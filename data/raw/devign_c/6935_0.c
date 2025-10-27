static void x86_cpu_parse_featurestr(const char *typename, char *features,

                                     Error **errp)

{

    char *featurestr; /* Single 'key=value" string being parsed */

    static bool cpu_globals_initialized;

    bool ambiguous = false;



    if (cpu_globals_initialized) {

        return;

    }

    cpu_globals_initialized = true;



    if (!features) {

        return;

    }



    for (featurestr = strtok(features, ",");

         featurestr;

         featurestr = strtok(NULL, ",")) {

        const char *name;

        const char *val = NULL;

        char *eq = NULL;

        char num[32];

        GlobalProperty *prop;



        /* Compatibility syntax: */

        if (featurestr[0] == '+') {

            plus_features = g_list_append(plus_features,

                                          g_strdup(featurestr + 1));

            continue;

        } else if (featurestr[0] == '-') {

            minus_features = g_list_append(minus_features,

                                           g_strdup(featurestr + 1));

            continue;

        }



        eq = strchr(featurestr, '=');

        if (eq) {

            *eq++ = 0;

            val = eq;

        } else {

            val = "on";

        }



        feat2prop(featurestr);

        name = featurestr;



        if (g_list_find_custom(plus_features, name, compare_string)) {

            error_report("warning: Ambiguous CPU model string. "

                         "Don't mix both \"+%s\" and \"%s=%s\"",

                         name, name, val);

            ambiguous = true;

        }

        if (g_list_find_custom(minus_features, name, compare_string)) {

            error_report("warning: Ambiguous CPU model string. "

                         "Don't mix both \"-%s\" and \"%s=%s\"",

                         name, name, val);

            ambiguous = true;

        }



        /* Special case: */

        if (!strcmp(name, "tsc-freq")) {

            int64_t tsc_freq;



            tsc_freq = qemu_strtosz_metric(val, NULL);

            if (tsc_freq < 0) {

                error_setg(errp, "bad numerical value %s", val);

                return;

            }

            snprintf(num, sizeof(num), "%" PRId64, tsc_freq);

            val = num;

            name = "tsc-frequency";

        }



        prop = g_new0(typeof(*prop), 1);

        prop->driver = typename;

        prop->property = g_strdup(name);

        prop->value = g_strdup(val);

        prop->errp = &error_fatal;

        qdev_prop_register_global(prop);

    }



    if (ambiguous) {

        error_report("warning: Compatibility of ambiguous CPU model "

                     "strings won't be kept on future QEMU versions");

    }

}
