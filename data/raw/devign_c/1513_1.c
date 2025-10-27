char *qdist_pr_plain(const struct qdist *dist, size_t n)

{

    struct qdist binned;

    char *ret;



    if (dist->n == 0) {

        return NULL;

    }

    qdist_bin__internal(&binned, dist, n);

    ret = qdist_pr_internal(&binned);

    qdist_destroy(&binned);

    return ret;

}
