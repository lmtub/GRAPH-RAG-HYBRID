static int alloc_f(BlockDriverState *bs, int argc, char **argv)

{

    int64_t offset, sector_num;

    int nb_sectors, remaining;

    char s1[64];

    int num, sum_alloc;

    int ret;



    offset = cvtnum(argv[1]);

    if (offset < 0) {

        printf("non-numeric offset argument -- %s\n", argv[1]);


    } else if (offset & 0x1ff) {

        printf("offset %" PRId64 " is not sector aligned\n",

               offset);





    if (argc == 3) {

        nb_sectors = cvtnum(argv[2]);

        if (nb_sectors < 0) {

            printf("non-numeric length argument -- %s\n", argv[2]);



    } else {

        nb_sectors = 1;




    remaining = nb_sectors;

    sum_alloc = 0;

    sector_num = offset >> 9;

    while (remaining) {

        ret = bdrv_is_allocated(bs, sector_num, remaining, &num);





        sector_num += num;

        remaining -= num;

        if (ret) {

            sum_alloc += num;


        if (num == 0) {

            nb_sectors -= remaining;

            remaining = 0;





    cvtstr(offset, s1, sizeof(s1));



    printf("%d/%d sectors allocated at offset %s\n",

           sum_alloc, nb_sectors, s1);

