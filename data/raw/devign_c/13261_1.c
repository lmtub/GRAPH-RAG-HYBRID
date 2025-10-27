static void set_alarm (m48t59_t *NVRAM, struct tm *tm)

{

    NVRAM->alarm = mktime(tm);

    if (NVRAM->alrm_timer != NULL) {

        qemu_del_timer(NVRAM->alrm_timer);

	NVRAM->alrm_timer = NULL;

    }

    if (NVRAM->alarm - time(NULL) > 0)

	qemu_mod_timer(NVRAM->alrm_timer, NVRAM->alarm * 1000);

}
