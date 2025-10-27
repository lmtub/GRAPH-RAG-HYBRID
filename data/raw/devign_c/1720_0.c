static void default_show_tags(WriterContext *wctx, AVDictionary *dict)

{

    AVDictionaryEntry *tag = NULL;

    while ((tag = av_dict_get(dict, "", tag, AV_DICT_IGNORE_SUFFIX))) {

        printf("TAG:");

        writer_print_string(wctx, tag->key, tag->value);

    }

}
