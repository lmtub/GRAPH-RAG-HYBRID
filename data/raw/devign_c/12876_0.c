static void channel_load_d(struct fs_dma_ctrl *ctrl, int c)

{

	target_phys_addr_t addr = channel_reg(ctrl, c, RW_SAVED_DATA);



	/* Load and decode. FIXME: handle endianness.  */

	D(printf("%s ch=%d addr=" TARGET_FMT_plx "\n", __func__, c, addr));

	cpu_physical_memory_read (addr,

				  (void *) &ctrl->channels[c].current_d, 

				  sizeof ctrl->channels[c].current_d);



	D(dump_d(c, &ctrl->channels[c].current_d));

	ctrl->channels[c].regs[RW_DATA] = addr;

}
