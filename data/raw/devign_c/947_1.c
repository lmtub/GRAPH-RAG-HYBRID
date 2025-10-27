int sws_getColorspaceDetails(SwsContext *c, int **inv_table, int *srcRange, int **table, int *dstRange, int *brightness, int *contrast, int *saturation)

{

    if (isYUV(c->dstFormat) || isGray(c->dstFormat)) return -1;



    *inv_table = c->srcColorspaceTable;

    *table     = c->dstColorspaceTable;

    *srcRange  = c->srcRange;

    *dstRange  = c->dstRange;

    *brightness= c->brightness;

    *contrast  = c->contrast;

    *saturation= c->saturation;



    return 0;

}
