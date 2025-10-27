void qemu_announce_self(void)

{

	static QEMUTimer *timer;

	timer = qemu_new_timer(rt_clock, qemu_announce_self_once, &timer);

	qemu_announce_self_once(&timer);

}
