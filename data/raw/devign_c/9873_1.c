vcard_emul_options(const char *args)

{

    int reader_count = 0;

    VCardEmulOptions *opts;



    /* Allow the future use of allocating the options structure on the fly */

    memcpy(&options, &default_options, sizeof(options));

    opts = &options;



    do {

        args = strip(args); /* strip off the leading spaces */

        if (*args == ',') {

            continue;

        }

        /* soft=(slot_name,virt_name,emul_type,emul_flags,cert_1, (no eol)

         *       cert_2,cert_3...) */

        if (strncmp(args, "soft=", 5) == 0) {

            const char *name;

            size_t name_length;

            const char *vname;

            size_t vname_length;

            const char *type_params;

            size_t type_params_length;

            char type_str[100];

            VCardEmulType type;

            int count, i;

            VirtualReaderOptions *vreaderOpt = NULL;



            args = strip(args + 5);

            if (*args != '(') {

                continue;

            }

            args = strip(args+1);



            NEXT_TOKEN(name)

            NEXT_TOKEN(vname)

            NEXT_TOKEN(type_params)

            type_params_length = MIN(type_params_length, sizeof(type_str)-1);

            memcpy(type_str, type_params, type_params_length);

            type_str[type_params_length] = '\0';

            type = vcard_emul_type_from_string(type_str);



            NEXT_TOKEN(type_params)



            if (*args == 0) {

                break;

            }



            if (opts->vreader_count >= reader_count) {

                reader_count += READER_STEP;

                vreaderOpt = g_renew(VirtualReaderOptions, opts->vreader,

                                     reader_count);

            }

            opts->vreader = vreaderOpt;

            vreaderOpt = &vreaderOpt[opts->vreader_count];

            vreaderOpt->name = g_strndup(name, name_length);

            vreaderOpt->vname = g_strndup(vname, vname_length);

            vreaderOpt->card_type = type;

            vreaderOpt->type_params =

                g_strndup(type_params, type_params_length);

            count = count_tokens(args, ',', ')') + 1;

            vreaderOpt->cert_count = count;

            vreaderOpt->cert_name = g_new(char *, count);

            for (i = 0; i < count; i++) {

                const char *cert = args;

                args = strpbrk(args, ",)");

                vreaderOpt->cert_name[i] = g_strndup(cert, args - cert);

                args = strip(args+1);

            }

            if (*args == ')') {

                args++;

            }

            opts->vreader_count++;

        /* use_hw= */

        } else if (strncmp(args, "use_hw=", 7) == 0) {

            args = strip(args+7);

            if (*args == '0' || *args == 'N' || *args == 'n' || *args == 'F') {

                opts->use_hw = PR_FALSE;

            } else {

                opts->use_hw = PR_TRUE;

            }

            args = find_blank(args);

        /* hw_type= */

        } else if (strncmp(args, "hw_type=", 8) == 0) {

            args = strip(args+8);

            opts->hw_card_type = vcard_emul_type_from_string(args);

            args = find_blank(args);

        /* hw_params= */

        } else if (strncmp(args, "hw_params=", 10) == 0) {

            const char *params;

            args = strip(args+10);

            params = args;

            args = find_blank(args);

            opts->hw_type_params = g_strndup(params, args-params);

        /* db="/data/base/path" */

        } else if (strncmp(args, "db=", 3) == 0) {

            const char *db;

            args = strip(args+3);

            if (*args != '"') {

                continue;

            }

            args++;

            db = args;

            args = strpbrk(args, "\"\n");

            opts->nss_db = g_strndup(db, args-db);

            if (*args != 0) {

                args++;

            }

        } else {

            args = find_blank(args);

        }

    } while (*args != 0);



    return opts;

}
