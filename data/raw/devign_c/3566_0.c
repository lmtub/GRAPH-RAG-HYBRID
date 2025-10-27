static int get_preset_file_2(const char *preset_name, const char *codec_name, AVIOContext **s)

{

    int i, ret = 1;

    char filename[1000];

    const char *base[3] = { getenv("AVCONV_DATADIR"),

                            getenv("HOME"),

                            AVCONV_DATADIR,

                            };



    for (i = 0; i < FF_ARRAY_ELEMS(base) && ret; i++) {

        if (!base[i])

            continue;

        if (codec_name) {

            snprintf(filename, sizeof(filename), "%s%s/%s-%s.avpreset", base[i],

                     i != 1 ? "" : "/.avconv", codec_name, preset_name);

            ret = avio_open2(s, filename, AVIO_FLAG_READ, &int_cb, NULL);

        }

        if (ret) {

            snprintf(filename, sizeof(filename), "%s%s/%s.avpreset", base[i],

                     i != 1 ? "" : "/.avconv", preset_name);

            ret = avio_open2(s, filename, AVIO_FLAG_READ, &int_cb, NULL);

        }

    }

    return ret;

}
