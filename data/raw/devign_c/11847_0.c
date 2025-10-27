void s390_feat_bitmap_to_ascii(const S390FeatBitmap bitmap, void *opaque,

                               void (*fn)(const char *name, void *opaque))

{

    S390Feat feat;



    feat = find_first_bit(bitmap, S390_FEAT_MAX);

    while (feat < S390_FEAT_MAX) {

        fn(s390_feat_def(feat)->name, opaque);

        feat = find_next_bit(bitmap, S390_FEAT_MAX, feat + 1);

    };

}
