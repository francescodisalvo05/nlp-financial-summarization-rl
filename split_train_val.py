"""Split train folders into two different ones: train, val"""
import argparse
import logging
import os

from sklearn.model_selection import train_test_split


def main(args):

    # collect all the train sample ids
    reports_ids = os.listdir(os.path.join(args.dataset_path,args.suffix,args.reports_folder))

    # split them into train and validation
    logging.info(f"Splitting {args.suffix}...")
    train_reports, validation_reports = train_test_split(reports_ids, test_size=0.15, random_state=42)

    # create val folder
    logging.info("Creating folders...")
    val_path = os.path.join(args.dataset_path,'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    os.makedirs(os.path.join(val_path,args.reports_folder))
    os.makedirs(os.path.join(val_path, args.summaries_folder))

    # move validation reports
    logging.info("Moving validation reports...")
    for report in validation_reports:
        src = os.path.join(args.dataset_path, args.suffix,args.reports_folder,report)
        dst = os.path.join(args.dataset_path, 'val', args.reports_folder, report)
        os.rename(src, dst)

    # move val summaries
    logging.info("Moving validation summaries...")
    for summary_filename in os.listdir(os.path.join(args.dataset_path,args.suffix,args.summaries_folder)):

        report_filename = summary_filename.split("_")[0] + '.txt'

        if report_filename in validation_reports:
            src = os.path.join(args.dataset_path, args.suffix, args.summaries_folder, summary_filename)
            dst = os.path.join(args.dataset_path, 'val', args.summaries_folder, summary_filename)
            os.rename(src, dst)

    # rename train into train
    logging.info(f"Renaming {args.suffix} into 'train'...")
    src = os.path.join(args.dataset_path, args.suffix)
    dst = os.path.join(args.dataset_path, 'train')
    os.rename(src,dst)

    logging.info(f"Completed.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-D', '--dataset_path', type=str, required=True, help='Set the base dataset path')
    parser.add_argument('-S', '--suffix', type=str, required=True, help='Set the trainval dataset suffix')
    parser.add_argument('-r', '--reports_folder', type=str, default='annual_reports', help='Name of the reports folder')
    parser.add_argument('-s', '--summaries_folder', type=str, default='gold_summaries', help='Name of the summaries folder')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    main(args)