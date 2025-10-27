static QString *read_line(FILE *file, char *key)

{

    char value[128];



    if (fscanf(file, "%s%s", key, value) == EOF)

        return NULL;

    remove_dots(key);

    return qstring_from_str(value);

}
