static int cow_is_allocated(BlockDriverState *bs, int64_t sector_num,

        int nb_sectors, int *num_same)

{

    int changed;



    if (nb_sectors == 0) {

	*num_same = nb_sectors;

	return 0;

    }



    changed = is_bit_set(bs, sector_num);

    if (changed < 0) {

        return 0; /* XXX: how to return I/O errors? */

    }



    for (*num_same = 1; *num_same < nb_sectors; (*num_same)++) {

	if (is_bit_set(bs, sector_num + *num_same) != changed)

	    break;

    }



    return changed;

}
