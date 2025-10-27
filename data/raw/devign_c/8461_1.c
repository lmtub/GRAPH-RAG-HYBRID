static int select_voice(cst_voice **voice, const char *voice_name, void *log_ctx)

{

    int i;



    for (i = 0; i < FF_ARRAY_ELEMS(voice_entries); i++) {

        struct voice_entry *entry = &voice_entries[i];

        if (!strcmp(entry->name, voice_name)) {

            *voice = entry->register_fn(NULL);

            if (!*voice) {

                av_log(log_ctx, AV_LOG_ERROR,

                       "Could not register voice '%s'\n", voice_name);

                return AVERROR_UNKNOWN;

            }

            return 0;

        }

    }



    av_log(log_ctx, AV_LOG_ERROR, "Could not find voice '%s'\n", voice_name);

    av_log(log_ctx, AV_LOG_INFO, "Choose between the voices: ");

    list_voices(log_ctx, ", ");



    return AVERROR(EINVAL);

}
