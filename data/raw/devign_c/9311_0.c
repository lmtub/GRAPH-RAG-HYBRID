static void logerr (struct audio_pt *pt, int err, const char *fmt, ...)

{

    va_list ap;



    va_start (ap, fmt);

    AUD_vlog (pt->drv, fmt, ap);

    va_end (ap);



    AUD_log (NULL, "\n");

    AUD_log (pt->drv, "Reason: %s\n", strerror (err));

}
