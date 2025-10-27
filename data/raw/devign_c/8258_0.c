static ImageInfoSpecific *qcow2_get_specific_info(BlockDriverState *bs)

{

    BDRVQcow2State *s = bs->opaque;

    ImageInfoSpecific *spec_info = g_new(ImageInfoSpecific, 1);



    *spec_info = (ImageInfoSpecific){

        .kind  = IMAGE_INFO_SPECIFIC_KIND_QCOW2,

        {

            .qcow2 = g_new(ImageInfoSpecificQCow2, 1),

        },

    };

    if (s->qcow_version == 2) {

        *spec_info->qcow2 = (ImageInfoSpecificQCow2){

            .compat             = g_strdup("0.10"),

            .refcount_bits      = s->refcount_bits,

        };

    } else if (s->qcow_version == 3) {

        *spec_info->qcow2 = (ImageInfoSpecificQCow2){

            .compat             = g_strdup("1.1"),

            .lazy_refcounts     = s->compatible_features &

                                  QCOW2_COMPAT_LAZY_REFCOUNTS,

            .has_lazy_refcounts = true,

            .corrupt            = s->incompatible_features &

                                  QCOW2_INCOMPAT_CORRUPT,

            .has_corrupt        = true,

            .refcount_bits      = s->refcount_bits,

        };

    }



    return spec_info;

}
