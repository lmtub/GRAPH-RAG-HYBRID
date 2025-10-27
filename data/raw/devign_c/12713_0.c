static int compare_sectors(const uint8_t *buf1, const uint8_t *buf2, int n,

    int *pnum)

{

    bool res;

    int i;



    if (n <= 0) {

        *pnum = 0;

        return 0;

    }



    res = !!memcmp(buf1, buf2, 512);

    for(i = 1; i < n; i++) {

        buf1 += 512;

        buf2 += 512;



        if (!!memcmp(buf1, buf2, 512) != res) {

            break;

        }

    }



    *pnum = i;

    return res;

}
