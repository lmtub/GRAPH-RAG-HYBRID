static void free_field_queue(PullupField *head, PullupField **last)

{

    PullupField *f = head;

    while (f) {

        av_free(f->diffs);

        av_free(f->combs);

        av_free(f->vars);

        if (f == *last) {

            av_freep(last);

            break;

        }

        f = f->next;

        av_freep(&f->prev);

    };

}
