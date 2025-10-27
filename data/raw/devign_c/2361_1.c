static bool key_is_missing(const BlockInfo *bdev)

{

    return (bdev->inserted && bdev->inserted->encryption_key_missing);

}
