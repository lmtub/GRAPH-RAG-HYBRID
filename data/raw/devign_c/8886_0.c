static inline void writer_print_string(WriterContext *wctx,

                                       const char *key, const char *val)

{

    wctx->writer->print_string(wctx, key, val);

    wctx->nb_item++;

}
