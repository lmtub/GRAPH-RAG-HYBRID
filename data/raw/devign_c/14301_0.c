void sws_freeFilter(SwsFilter *filter)

{

    if (!filter)

        return;



    if (filter->lumH)

        sws_freeVec(filter->lumH);

    if (filter->lumV)

        sws_freeVec(filter->lumV);

    if (filter->chrH)

        sws_freeVec(filter->chrH);

    if (filter->chrV)

        sws_freeVec(filter->chrV);

    av_free(filter);

}
