void ff_id3v2_free_extra_meta(ID3v2ExtraMeta **extra_meta)

{

    ID3v2ExtraMeta *current = *extra_meta, *next;

    void (*free_func)(ID3v2ExtraMeta*);



    while (current) {

        if ((free_func = get_extra_meta_func(current->tag, 1)->free))

            free_func(current->data);

        next = current->next;

        av_freep(&current);

        current = next;

    }

}
