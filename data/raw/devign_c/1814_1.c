static int parse_key(DBEContext *s)

{

    int key = 0;



    if (s->key_present && s->input_size > 0)

        key = AV_RB24(s->input) >> 24 - s->word_bits;



    skip_input(s, s->key_present);

    return key;

}
