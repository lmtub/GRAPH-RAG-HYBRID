static void omap_rtc_alarm_update(struct omap_rtc_s *s)

{

    s->alarm_ti = mktimegm(&s->alarm_tm);

    if (s->alarm_ti == -1)

        printf("%s: conversion failed\n", __FUNCTION__);

}
