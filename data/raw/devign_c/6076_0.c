static void channel_store_c(struct fs_dma_ctrl *ctrl, int c)

{

	target_phys_addr_t addr = channel_reg(ctrl, c, RW_GROUP_DOWN);



	/* Encode and store. FIXME: handle endianness.  */

	D(printf("%s ch=%d addr=" TARGET_FMT_plx "\n", __func__, c, addr));

	D(dump_d(c, &ctrl->channels[c].current_d));

	cpu_physical_memory_write (addr,

				  (void *) &ctrl->channels[c].current_c,

				  sizeof ctrl->channels[c].current_c);

}
