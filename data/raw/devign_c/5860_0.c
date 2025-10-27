real_parse_asm_rule(AVStream *st, const char *p, const char *end)

{

    do {

        /* can be either averagebandwidth= or AverageBandwidth= */

#if AV_HAVE_INCOMPATIBLE_LIBAV_ABI

        if (sscanf(p, " %*1[Aa]verage%*1[Bb]andwidth=%d", &st->codec->bit_rate) == 1)

#else

        if (sscanf(p, " %*1[Aa]verage%*1[Bb]andwidth=%"SCNd64, &st->codec->bit_rate) == 1)

#endif

            break;

        if (!(p = strchr(p, ',')) || p > end)

            p = end;

        p++;

    } while (p < end);

}
