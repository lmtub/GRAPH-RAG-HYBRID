static int rtc_start_timer(struct qemu_alarm_timer *t)

{

    int rtc_fd;

    unsigned long current_rtc_freq = 0;



    TFR(rtc_fd = open("/dev/rtc", O_RDONLY));

    if (rtc_fd < 0)

        return -1;

    ioctl(rtc_fd, RTC_IRQP_READ, &current_rtc_freq);

    if (current_rtc_freq != RTC_FREQ &&

        ioctl(rtc_fd, RTC_IRQP_SET, RTC_FREQ) < 0) {

        fprintf(stderr, "Could not configure '/dev/rtc' to have a 1024 Hz timer. This is not a fatal\n"

                "error, but for better emulation accuracy either use a 2.6 host Linux kernel or\n"

                "type 'echo 1024 > /proc/sys/dev/rtc/max-user-freq' as root.\n");

        goto fail;

    }

    if (ioctl(rtc_fd, RTC_PIE_ON, 0) < 0) {

    fail:

        close(rtc_fd);

        return -1;

    }



    enable_sigio_timer(rtc_fd);



    t->priv = (void *)(long)rtc_fd;



    return 0;

}
