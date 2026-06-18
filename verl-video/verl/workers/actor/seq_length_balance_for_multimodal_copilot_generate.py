// ...existing code...
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        # batch_td_for_dataloader will be a TensorDict containing only the select_keys
        batch_td_for_dataloader = data.batch.select(*select_keys, strict=False)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            # Create a DataProto for dataloader that contains selected tensor keys and multi_modal_inputs
            # This ensures each mini_batch from dataloader has the necessary structure
            non_tensor_select_keys = ["multi_modal_inputs"]
            # We need to ensure that the DataProto passed to chunk also has its .batch attribute
            # correctly reflecting only the selected tensor keys.
            # Create a temporary DataProto for chunking if necessary, or ensure select() handles this.
            # data.select() returns a new DataProto with selected fields.
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            # For text-only, dataloader iterates over TensorDicts
            dataloader = batch_td_for_dataloader.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch_data_item in enumerate(dataloader): # mini_batch_data_item is DataProto or TensorDict
                # split batch into micro_batches
                micro_batches = []
                if has_multi_modal_inputs: # mini_batch_data_item is DataProto here
                    # mini_batch_data_item.batch contains the selected tensors
                    # mini_batch_data_item.non_tensor_batch contains {"multi_modal_inputs": ...}
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = mini_batch_data_item.non_tensor_batch["multi_modal_inputs"]
                        # batch_tensordict_for_rearrange contains only the selected textual/tensor keys for this mini_batch
                        batch_tensordict_for_rearrange = mini_batch_data_item.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size

                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for i, text_mb_td in enumerate(rearranged_text_micro_batches_tds):
                            current_original_indices = textual_indices[i]
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else: # Original non-dynamic multimodal logic
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch_data_item.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        # mini_batch_data_item already contains selected keys, so chunking it is fine.
                        micro_batches = mini_batch_data_item.chunk(num_micro_batches) # Returns List[DataProto]
                elif self.config.use_dynamic_bsz: # Text-only, dynamic bsz. mini_batch_data_item is TensorDict.
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    # mini_batch_data_item is already the TensorDict for the current PPO mini-batch
                    rearranged_tds, _ = rearrange_micro_batches(batch=mini_batch_data_item, max_token_len=max_token_len)
                    micro_batches = rearranged_tds # List of TensorDicts or dicts
                else: # Text-only, non-dynamic bsz. mini_batch_data_item is TensorDict.
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch_data_item.split(self.config.ppo_micro_batch_size_per_gpu) # List of TensorDicts

                self.actor_optimizer.zero_grad()

                for micro_batch_item in micro_batches:
                    current_micro_batch_data_on_device = {}
                    if isinstance(micro_batch_item, DataProto): # Non-dynamic multimodal case
                        for k, v in micro_batch_item.batch.items(): # batch is TensorDict
                            current_micro_batch_data_on_device[k] = v.to(torch.cuda.current_device())
                        for k, v in micro_batch_item.non_tensor_batch.items():
                            if k == "multi_modal_inputs" and v is not None:
                                current_micro_batch_data_on_device[k] = [
                                    {kk: vv.to(torch.cuda.current_device()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else:
                                current_micro_batch_data_on_device[k] = v
                    elif isinstance(micro_batch_item, dict): # Dynamic multimodal (new) or text-only dynamic (if rearrange_micro_batches returns dicts)
                        for k, v in micro_batch_item.items():
                            if isinstance(v, torch.Tensor):
                                current_micro_batch_data_on_device[k] = v.to(torch.cuda.current_device())
                            elif k == "multi_modal_inputs" and v is not None: # v is List[Dict[str, Tensor]]
                                current_micro_batch_data_on_device[k] = [
                                    {kk: vv.to(torch.cuda.current_device()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else: # Other non-tensor data (e.g. list of strings, if any)
                                current_micro_batch_data_on_device[k] = v
                    else: # Assuming TensorDict for text-only cases
                          # (non-dynamic or dynamic if rearrange_micro_batches returns TensorDicts)
                        # .to(device) handles all tensors in TensorDict
                        td_on_device = micro_batch_item.to(torch.cuda.current_device())
                        # _forward_micro_batch expects a dict-like interface
                        current_micro_batch_data_on_device = td_on_device

                    responses = current_micro_batch_data_on_device["responses"]
                    response_length = responses.size(1)
                    attention_mask = current_micro_batch_data_on_device["attention_mask"]
                    if multi_turn:
                        response_mask = current_micro_batch_data_on_device["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = current_micro_batch_data_on_device["old_log_probs"]
                    advantages = current_micro_batch_data_on_device["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=current_micro_batch_data_on_device,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy
                    )

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=// filepath: /cpfs01/user/wangzerui/verl/verl/workers/actor/dp_actor.py
// ...existing code...
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        # batch_td_for_dataloader will be a TensorDict containing only the select_keys
        batch_td_for_dataloader = data.batch.select(*select_keys, strict=False)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            # Create a DataProto for dataloader that contains selected tensor keys and multi_modal_inputs
            # This ensures each mini_batch from dataloader has the necessary structure
            non_tensor_select_keys = ["multi_modal_inputs"]
            # We need to ensure that the DataProto passed to chunk also has its .batch attribute
            # correctly reflecting only the selected tensor keys.
            # Create a temporary DataProto for chunking if necessary, or ensure select() handles this.
            # data.select() returns a new DataProto with selected fields.
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            # For text-only, dataloader iterates over TensorDicts
            dataloader = batch_td_for_dataloader.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch_data_item in enumerate(dataloader): # mini_batch_data_item is DataProto or TensorDict
                # split batch into micro_batches
                micro_batches = []
                if has_multi_modal_inputs: # mini_batch_data_item is DataProto here
                    # mini_batch_data_item.batch contains the selected tensors
                    # mini_batch_data_item.non_tensor_batch contains {"multi_modal_inputs": ...}
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = mini_batch_data_item.non_tensor_batch["multi_modal_inputs"]
                        # batch_tensordict_for_rearrange contains only the selected textual/tensor keys for this mini_batch
                        batch_tensordict_for_rearrange = mini_batch_data_item.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size

                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for i, text_mb_td in enumerate(rearranged_text_micro_batches_tds):
                            current_original_indices = textual_indices[i]
                            current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else: # Original non-dynamic multimodal logic
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch_data_item.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        # mini_batch_data_item already contains selected keys, so chunking it is fine.
                        micro_batches = mini_batch_data_item.chunk(num_micro_batches) # Returns List[DataProto]
                elif self.config.use_dynamic_bsz: # Text-only, dynamic bsz. mini_batch_data_item is TensorDict.
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    # mini_batch_data_item is already the TensorDict for the current PPO mini-batch
                    rearranged_tds, _ = rearrange_micro_batches(batch=mini_batch_data_item, max_token_len=max_token_len)
                    micro_batches = rearranged_tds # List of TensorDicts or dicts
                else: # Text-only, non-dynamic bsz. mini_batch_data_item is TensorDict.
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch_data_item.split(self.config.ppo_micro_batch_size_per_gpu) # List of TensorDicts

                self.actor_optimizer.zero_grad()

                for micro_batch_item in micro_batches:
                    current_micro_batch_data_on_device = {}
                    if isinstance(micro_batch_item, DataProto): # Non-dynamic multimodal case
                        for k, v in micro_batch_item.batch.items(): # batch is TensorDict
                            current_micro_batch_data_on_device[k] = v.to(torch.cuda.current_device())
                        for k, v in micro_batch_item.non_tensor_batch.items():
                            if k == "multi_modal_inputs" and v is not None:
                                current_micro_batch_data_on_device[k] = [
                                    {kk: vv.to(torch.cuda.current_device()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else:
                                current_micro_batch_data_on_device[k] = v
                    elif isinstance(micro_batch_item, dict): # Dynamic multimodal (new) or text-only dynamic (if rearrange_micro_batches returns dicts)
                        for k, v in micro_batch_item.items():
                            if isinstance(v, torch.Tensor):
                                current_micro_batch_data_on_device[k] = v.to(torch.cuda.current_device())
                            elif k == "multi_modal_inputs" and v is not None: # v is List[Dict[str, Tensor]]
                                current_micro_batch_data_on_device[k] = [
                                    {kk: vv.to(torch.cuda.current_device()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else: # Other non-tensor data (e.g. list of strings, if any)
                                current_micro_batch_data_on_device[k] = v
                    else: # Assuming TensorDict for text-only cases
                          # (non-dynamic or dynamic if rearrange_micro_batches returns TensorDicts)
                        # .to(device) handles all tensors in TensorDict
                        td_on_device = micro_batch_item.to(torch.cuda.current_device())
                        # _forward_micro_batch expects a dict-like interface
                        current_micro_batch_data_on_device = td_on_device

                    responses = current_micro_batch_data_on_device["responses"]
                    response_length = responses.size(1)
                    attention_mask = current_micro_batch_data_on_device["attention_mask"]
                    if multi_turn:
                        response_mask = current_micro_batch_data_on_device["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = current_micro_batch_data_on_device["old_log_probs"]
                    advantages = current_micro_batch_data_on_device["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True

                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=current_micro_batch_data_on_device,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy
                    )

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=