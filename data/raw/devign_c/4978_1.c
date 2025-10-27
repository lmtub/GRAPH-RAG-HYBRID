ReadLineState *readline_init(ReadLinePrintfFunc *printf_func,

                             ReadLineFlushFunc *flush_func,

                             void *opaque,

                             ReadLineCompletionFunc *completion_finder)

{

    ReadLineState *rs = g_malloc0(sizeof(*rs));



    rs->hist_entry = -1;

    rs->opaque = opaque;

    rs->printf_func = printf_func;

    rs->flush_func = flush_func;

    rs->completion_finder = completion_finder;



    return rs;

}
