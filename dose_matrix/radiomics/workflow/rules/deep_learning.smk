
# Normalize the image into on
rule images_nii_dl:
    input:
        get_newdosi_files,
        METADATA_DIR + "biggest_image_size.csv"
    output:
        NII_DL_DIR + "{newdosi_patient}_ID2013A.nii.gz"
    run:
        df_size = pd.read_csv(METADATA_DIR + "biggest_image_size.csv", index_col = 0, names = ["size"], header = None)
        biggest_image_size = df_size.loc[["size_x", "size_y", "size_z"], "size"].values
        logger = setup_logger("csv2nii_dl", NII_DL_DIR + "csv2nii_dl.log", mode_file = "a", creation_msg = False)
        logger.info(f"{wildcards.newdosi_patient}: creation of nii images")
        subdir = os.path.dirname(wildcards.newdosi_patient)
        list_newdosi_files_patient = get_newdosi_files(wildcards)
        list_filenames = [os.path.basename(newdosi_file) for newdosi_file in list_newdosi_files_patient]
        path_csv = DOSES_DATASET_DIR + subdir + "/"
        path_nii = NII_DL_DIR + subdir + "/"
        csv2nii.to_nii(path_csv, path_nii, list_filenames, NAME_SUPER_T_FUNC,
                       biggest_image_size = biggest_image_size, save_masks = False, save_empty = True)

rule end_images_nii_dl:
    input:
        expand(NII_DL_DIR + "{newdosi_patient}_ID2013A.nii.gz", newdosi_patient = list_newdosi_patients)
    output:
        NII_DL_DIR + "end_csv2nii_dl.log"
    run:
        with open(NII_DL_DIR + "end_csv2nii_dl.log", "w") as logfile:
            logfile.write(f"End of {len(list_newdosi_patients)} dose images treatments.")

