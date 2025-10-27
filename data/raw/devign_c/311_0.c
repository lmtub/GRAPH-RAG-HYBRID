bool error_is_type(Error *err, const char *fmt)

{

    const char *error_class;

    char *ptr;

    char *end;



    if (!err) {

        return false;

    }



    ptr = strstr(fmt, "'class': '");

    assert(ptr != NULL);

    ptr += strlen("'class': '");



    end = strchr(ptr, '\'');

    assert(end != NULL);



    error_class = error_get_field(err, "class");

    if (strlen(error_class) != end - ptr) {

        return false;

    }



    return strncmp(ptr, error_class, end - ptr) == 0;

}
