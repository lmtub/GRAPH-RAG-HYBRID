static char *value_string(char *buf, int buf_size, struct unit_value uv)

{

    double vald;

    int show_float = 0;



    if (uv.unit == unit_second_str) {

        vald = uv.val.d;

        show_float = 1;

    } else {

        vald = uv.val.i;

    }



    if (uv.unit == unit_second_str && use_value_sexagesimal_format) {

        double secs;

        int hours, mins;

        secs  = vald;

        mins  = (int)secs / 60;

        secs  = secs - mins * 60;

        hours = mins / 60;

        mins %= 60;

        snprintf(buf, buf_size, "%d:%02d:%09.6f", hours, mins, secs);

    } else {

        const char *prefix_string = "";

        int l;



        if (use_value_prefix && vald > 1) {

            long long int index;



            if (uv.unit == unit_byte_str && use_byte_value_binary_prefix) {

                index = (long long int) (log2(vald)) / 10;

                index = av_clip(index, 0, FF_ARRAY_ELEMS(binary_unit_prefixes) - 1);

                vald /= exp2(index * 10);

                prefix_string = binary_unit_prefixes[index];

            } else {

                index = (long long int) (log10(vald)) / 3;

                index = av_clip(index, 0, FF_ARRAY_ELEMS(decimal_unit_prefixes) - 1);

                vald /= pow(10, index * 3);

                prefix_string = decimal_unit_prefixes[index];

            }

        }



        if (show_float || (use_value_prefix && vald != (long long int)vald))

            l = snprintf(buf, buf_size, "%f", vald);

        else

            l = snprintf(buf, buf_size, "%lld", (long long int)vald);

        snprintf(buf+l, buf_size-l, "%s%s%s", *prefix_string || show_value_unit ? " " : "",

                 prefix_string, show_value_unit ? uv.unit : "");

    }



    return buf;

}
