static void channel_store_d(struct fs_dma_ctrl *ctrl, int c)

{

	target_phys_addr_t addr = channel_reg(ctrl, c, RW_SAVED_DATA);



	/* Encode and store. FIXME: handle endianness.  */

	D(printf("%s ch=%d addr=" TARGET_FMT_plx "\n", __func__, c, addr));

	cpu_physical_memory_write (addr,

				  (void *) &ctrl->channels[c].current_d, 

				  sizeof ctrl->channels[c].current_d);

}
