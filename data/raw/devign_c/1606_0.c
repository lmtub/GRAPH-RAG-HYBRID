int Configure(void **ctxp, int argc, char *argv[])

{

    ContextInfo *ci;

    int c;



    *ctxp = av_mallocz(sizeof(ContextInfo));

    ci = (ContextInfo *) *ctxp;



    optind = 1;



    ci->dir = "/tmp";

    ci->threshold = 100;

    ci->file_limit = 100;

    ci->min_interval = 1000000;

    ci->inset = 10;     /* Percent */



    while ((c = getopt(argc, argv, "w:i:dh:s:v:zl:t:D:")) > 0) {

        switch (c) {

            case 'h':

                dorange(optarg, &ci->dark.h, &ci->bright.h, 360);

                break;

            case 's':

                dorange(optarg, &ci->dark.s, &ci->bright.s, 255);

                break;

            case 'v':

                dorange(optarg, &ci->dark.v, &ci->bright.v, 255);

                break;

            case 'z':

                ci->zapping = 1;

                break;

            case 'l':

                ci->file_limit = atoi(optarg);

                break;

            case 'i':

                ci->min_interval = 1000000 * atof(optarg);

                break;

            case 't':

                ci->threshold = atof(optarg) * 1000;

                if (ci->threshold > 1000 || ci->threshold < 0) {

                    av_log(NULL, AV_LOG_ERROR, "Invalid threshold value '%s' (range is 0-1)\n", optarg);

                    return -1;

                }

                break;

            case 'w':

                ci->min_width = atoi(optarg);

                break;

            case 'd':

                ci->debug++;

                break;

            case 'D':

                ci->dir = av_strdup(optarg);

                break;

            default:

                av_log(NULL, AV_LOG_ERROR, "Unrecognized argument '%s'\n", argv[optind]);

                return -1;

        }

    }



    av_log(NULL, AV_LOG_INFO, "Fish detector configured:\n");

    av_log(NULL, AV_LOG_INFO, "    HSV range: %d,%d,%d - %d,%d,%d\n",

                        ci->dark.h,

                        ci->dark.s,

                        ci->dark.v,

                        ci->bright.h,

                        ci->bright.s,

                        ci->bright.v);

    av_log(NULL, AV_LOG_INFO, "    Threshold is %d%% pixels\n", ci->threshold / 10);





    return 0;

}
