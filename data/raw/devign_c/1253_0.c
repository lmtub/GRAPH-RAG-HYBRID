static void add_flagname_to_bitmaps(const char *flagname, uint32_t *features,

                                    uint32_t *ext_features,

                                    uint32_t *ext2_features,

                                    uint32_t *ext3_features,

                                    uint32_t *kvm_features,

                                    uint32_t *svm_features)

{

    if (!lookup_feature(features, flagname, NULL, feature_name) &&

        !lookup_feature(ext_features, flagname, NULL, ext_feature_name) &&

        !lookup_feature(ext2_features, flagname, NULL, ext2_feature_name) &&

        !lookup_feature(ext3_features, flagname, NULL, ext3_feature_name) &&

        !lookup_feature(kvm_features, flagname, NULL, kvm_feature_name) &&

        !lookup_feature(svm_features, flagname, NULL, svm_feature_name))

            fprintf(stderr, "CPU feature %s not found\n", flagname);

}
