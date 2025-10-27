static void *oss_audio_init (void)

{

    OSSConf *conf = g_malloc(sizeof(OSSConf));

    *conf = glob_conf;



    if (access(conf->devpath_in, R_OK | W_OK) < 0 ||

        access(conf->devpath_out, R_OK | W_OK) < 0) {


        return NULL;

    }

    return conf;

}