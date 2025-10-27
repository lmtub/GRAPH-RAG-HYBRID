int rtsp_next_attr_and_value(const char **p, char *attr, int attr_size, char *value, int value_size)

{

    skip_spaces(p);

    if(**p) {

        get_word_sep(attr, attr_size, "=", p);

        if (**p == '=')

            (*p)++;

        get_word_sep(value, value_size, ";", p);

        if (**p == ';')

            (*p)++;

        return 1;

    }

    return 0;

}
