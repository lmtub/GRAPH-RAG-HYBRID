static void channel_load_c(struct fs_dma_ctrl *ctrl, int c)

{

	target_phys_addr_t addr = channel_reg(ctrl, c, RW_GROUP_DOWN);



	/* Load and decode. FIXME: handle endianness.  */

	cpu_physical_memory_read (addr, 

				  (void *) &ctrl->channels[c].current_c, 

				  sizeof ctrl->channels[c].current_c);



	D(dump_c(c, &ctrl->channels[c].current_c));

	/* I guess this should update the current pos.  */

	ctrl->channels[c].regs[RW_SAVED_DATA] =

		(uint32_t)(unsigned long)ctrl->channels[c].current_c.saved_data;

	ctrl->channels[c].regs[RW_SAVED_DATA_BUF] =

		(uint32_t)(unsigned long)ctrl->channels[c].current_c.saved_data_buf;

}
