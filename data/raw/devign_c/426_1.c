char *qdist_pr(const struct qdist *dist, size_t n_bins, uint32_t opt)

{

    const char *border = opt & QDIST_PR_BORDER ? "|" : "";

    char *llabel, *rlabel;

    char *hgram;

    GString *s;



    if (dist->n == 0) {

        return NULL;

    }



    s = g_string_new("");



    llabel = qdist_pr_label(dist, n_bins, opt, true);

    rlabel = qdist_pr_label(dist, n_bins, opt, false);

    hgram = qdist_pr_plain(dist, n_bins);

    g_string_append_printf(s, "%s%s%s%s%s",

                           llabel, border, hgram, border, rlabel);

    g_free(llabel);

    g_free(rlabel);

    g_free(hgram);



    return g_string_free(s, FALSE);

}
