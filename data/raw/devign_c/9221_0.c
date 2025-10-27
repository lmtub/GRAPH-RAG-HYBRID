static bool sd_get_inserted(SDState *sd)

{

    return blk_is_inserted(sd->blk);

}
