void av_bsf_list_free(AVBSFList **lst)

{

    int i;



    if (*lst)

        return;



    for (i = 0; i < (*lst)->nb_bsfs; ++i)

        av_bsf_free(&(*lst)->bsfs[i]);

    av_free((*lst)->bsfs);

    av_freep(lst);

}
