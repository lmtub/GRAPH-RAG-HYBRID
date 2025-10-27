static void channel_out_run(struct fs_dma_ctrl *ctrl, int c)

{

	uint32_t len;

	uint32_t saved_data_buf;

	unsigned char buf[2 * 1024];



	if (ctrl->channels[c].eol == 1)

		return;



	saved_data_buf = channel_reg(ctrl, c, RW_SAVED_DATA_BUF);



	D(fprintf(logfile, "ch=%d buf=%x after=%x saved_data_buf=%x\n",

		 c,

		 (uint32_t)ctrl->channels[c].current_d.buf,

		 (uint32_t)ctrl->channels[c].current_d.after,

		 saved_data_buf));



	len = (uint32_t)(unsigned long) ctrl->channels[c].current_d.after;

	len -= saved_data_buf;



	if (len > sizeof buf)

		len = sizeof buf;

	cpu_physical_memory_read (saved_data_buf, buf, len);



	D(printf("channel %d pushes %x %u bytes\n", c, 

		 saved_data_buf, len));



	if (ctrl->channels[c].client->client.push)

		ctrl->channels[c].client->client.push(

			ctrl->channels[c].client->client.opaque, buf, len);

	else

		printf("WARNING: DMA ch%d dataloss, no attached client.\n", c);



	saved_data_buf += len;



	if (saved_data_buf ==

	    (uint32_t)(unsigned long)ctrl->channels[c].current_d.after) {

		/* Done. Step to next.  */

		if (ctrl->channels[c].current_d.out_eop) {

			/* TODO: signal eop to the client.  */

			D(printf("signal eop\n"));

		}

		if (ctrl->channels[c].current_d.intr) {

			/* TODO: signal eop to the client.  */

			/* data intr.  */

			D(printf("signal intr\n"));

			ctrl->channels[c].regs[R_INTR] |= (1 << 2);

			channel_update_irq(ctrl, c);

		}

		if (ctrl->channels[c].current_d.eol) {

			D(printf("channel %d EOL\n", c));

			ctrl->channels[c].eol = 1;



			/* Mark the context as disabled.  */

			ctrl->channels[c].current_c.dis = 1;

			channel_store_c(ctrl, c);



			channel_stop(ctrl, c);

		} else {

			ctrl->channels[c].regs[RW_SAVED_DATA] =

				(uint32_t)(unsigned long) ctrl->channels[c].current_d.next;

			/* Load new descriptor.  */

			channel_load_d(ctrl, c);

			saved_data_buf = (uint32_t)(unsigned long)

				ctrl->channels[c].current_d.buf;

		}



		channel_store_d(ctrl, c);

		ctrl->channels[c].regs[RW_SAVED_DATA_BUF] = saved_data_buf;

		D(dump_d(c, &ctrl->channels[c].current_d));

	}

	ctrl->channels[c].regs[RW_SAVED_DATA_BUF] = saved_data_buf;

}
