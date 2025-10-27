static av_cold int bktr_init(const char *video_device, int width, int height,

    int format, int *video_fd, int *tuner_fd, int idev, double frequency)

{

    struct meteor_geomet geo;

    int h_max;

    long ioctl_frequency;

    char *arg;

    int c;

    struct sigaction act = { 0 }, old;



    if (idev < 0 || idev > 4)

    {

        arg = getenv ("BKTR_DEV");

        if (arg)

            idev = atoi (arg);

        if (idev < 0 || idev > 4)

            idev = 1;

    }



    if (format < 1 || format > 6)

    {

        arg = getenv ("BKTR_FORMAT");

        if (arg)

            format = atoi (arg);

        if (format < 1 || format > 6)

            format = VIDEO_FORMAT;

    }



    if (frequency <= 0)

    {

        arg = getenv ("BKTR_FREQUENCY");

        if (arg)

            frequency = atof (arg);

        if (frequency <= 0)

            frequency = 0.0;

    }



    sigemptyset(&act.sa_mask);

    act.sa_handler = catchsignal;

    sigaction(SIGUSR1, &act, &old);



    *tuner_fd = open("/dev/tuner0", O_RDONLY);

    if (*tuner_fd < 0)

        av_log(NULL, AV_LOG_ERROR, "Warning. Tuner not opened, continuing: %s\n", strerror(errno));



    *video_fd = open(video_device, O_RDONLY);

    if (*video_fd < 0) {

        av_log(NULL, AV_LOG_ERROR, "%s: %s\n", video_device, strerror(errno));

        return -1;

    }



    geo.rows = height;

    geo.columns = width;

    geo.frames = 1;

    geo.oformat = METEOR_GEO_YUV_422 | METEOR_GEO_YUV_12;



    switch (format) {

    case PAL:   h_max = PAL_HEIGHT;   c = BT848_IFORM_F_PALBDGHI; break;

    case PALN:  h_max = PAL_HEIGHT;   c = BT848_IFORM_F_PALN;     break;

    case PALM:  h_max = PAL_HEIGHT;   c = BT848_IFORM_F_PALM;     break;

    case SECAM: h_max = SECAM_HEIGHT; c = BT848_IFORM_F_SECAM;    break;

    case NTSC:  h_max = NTSC_HEIGHT;  c = BT848_IFORM_F_NTSCM;    break;

    case NTSCJ: h_max = NTSC_HEIGHT;  c = BT848_IFORM_F_NTSCJ;    break;

    default:    h_max = PAL_HEIGHT;   c = BT848_IFORM_F_PALBDGHI; break;

    }



    if (height <= h_max / 2)

        geo.oformat |= METEOR_GEO_EVEN_ONLY;



    if (ioctl(*video_fd, METEORSETGEO, &geo) < 0) {

        av_log(NULL, AV_LOG_ERROR, "METEORSETGEO: %s\n", strerror(errno));

        return -1;

    }



    if (ioctl(*video_fd, BT848SFMT, &c) < 0) {

        av_log(NULL, AV_LOG_ERROR, "BT848SFMT: %s\n", strerror(errno));

        return -1;

    }



    c = bktr_dev[idev];

    if (ioctl(*video_fd, METEORSINPUT, &c) < 0) {

        av_log(NULL, AV_LOG_ERROR, "METEORSINPUT: %s\n", strerror(errno));

        return -1;

    }



    video_buf_size = width * height * 12 / 8;



    video_buf = (uint8_t *)mmap((caddr_t)0, video_buf_size,

        PROT_READ, MAP_SHARED, *video_fd, (off_t)0);

    if (video_buf == MAP_FAILED) {

        av_log(NULL, AV_LOG_ERROR, "mmap: %s\n", strerror(errno));

        return -1;

    }



    if (frequency != 0.0) {

        ioctl_frequency  = (unsigned long)(frequency*16);

        if (ioctl(*tuner_fd, TVTUNER_SETFREQ, &ioctl_frequency) < 0)

            av_log(NULL, AV_LOG_ERROR, "TVTUNER_SETFREQ: %s\n", strerror(errno));

    }



    c = AUDIO_UNMUTE;

    if (ioctl(*tuner_fd, BT848_SAUDIO, &c) < 0)

        av_log(NULL, AV_LOG_ERROR, "TVTUNER_SAUDIO: %s\n", strerror(errno));



    c = METEOR_CAP_CONTINOUS;

    ioctl(*video_fd, METEORCAPTUR, &c);



    c = SIGUSR1;

    ioctl(*video_fd, METEORSSIGNAL, &c);



    return 0;

}
