static int parse_metadata_ext(DBEContext *s)

{

    if (s->mtd_ext_size)

        skip_input(s, s->key_present + s->mtd_ext_size + 1);

    return 0;

}
