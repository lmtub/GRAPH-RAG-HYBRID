static void set_up_watchdog (m48t59_t *NVRAM, uint8_t value)

{

    uint64_t interval; /* in 1/16 seconds */



    if (NVRAM->wd_timer != NULL) {

        qemu_del_timer(NVRAM->wd_timer);

	NVRAM->wd_timer = NULL;

    }

    NVRAM->buffer[0x1FF0] &= ~0x80;

    if (value != 0) {

	interval = (1 << (2 * (value & 0x03))) * ((value >> 2) & 0x1F);

	qemu_mod_timer(NVRAM->wd_timer, ((uint64_t)time(NULL) * 1000) +

		       ((interval * 1000) >> 4));

    }

}
