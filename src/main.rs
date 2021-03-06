#![allow(non_snake_case)]

use ndarray::{Array1, Array2, Axis, Slice};
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::time::SystemTime;
use std::env;

use Enarx_Demo::Tree;


#[derive(Debug, Deserialize)]
struct DiabetesRecord {
    Pregnancies: f64,
    Glucose: f64,
    BloodPressure: f64,
    SkinThickness: f64,
    Insulin: f64,
    Bmi: f64,
    DiabetesPedigreeFunction: f64,
    Age: f64,
    Outcome: f64,
}

struct DiabetesDataSet {
    features: Array2<f64>,
    labels: Array1<f64>,
}

impl DiabetesDataSet {
    pub fn from(records: Vec<DiabetesRecord>) -> Self {
        let mut features = Vec::with_capacity(records.len());
        let mut labels = Vec::with_capacity(records.len());

        for r in records {
            let feats = &[
                r.Pregnancies, r.Glucose, r.BloodPressure, r.SkinThickness, r.Insulin, r.Bmi, r.DiabetesPedigreeFunction, r.Age,
            ];
            features.push(feats.clone());
            labels.push(r.Outcome);
        }

        Self {
            features: Array2::from(features),
            labels: Array1::from(labels),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {

    let args: Vec<String> = env::args().collect();


    println!("external file path {:?}", args[1]);
   

    let file_path = &args[1];
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut bfr = Vec::new();

    for result in rdr.deserialize() {
        let record: DiabetesRecord = result?;
        bfr.push(record);
    }

    let dataset = DiabetesDataSet::from(bfr);
    println!("Dataset size: {}", &dataset.features.nrows());

    let train_dataset = DiabetesDataSet {
        features: dataset
            .features
            .slice_axis(Axis(0), Slice::from(0..450))
            .to_owned(),
        labels: dataset
            .labels
            .slice_axis(Axis(0), Slice::from(0..450))
            .to_owned(),
    };

    let test_dataset = DiabetesDataSet {
        features: dataset
            .features
            .slice_axis(Axis(0), Slice::from(450..))
            .to_owned(),
        labels: dataset
            .labels
            .slice_axis(Axis(0), Slice::from(450..))
            .to_owned(),
    };

    let start_training = SystemTime::now();
    let decision_tree = Tree::new(
        train_dataset.features.clone(),
        train_dataset.labels.clone(),
        5,
        7,
    );
    let end_training = start_training.elapsed().unwrap().as_millis();
    println!("Training took {} ms", end_training);

    let start_prediction = SystemTime::now();
    let yhat_train = train_dataset
        .features
        .map_axis(Axis(1), |x| decision_tree.predict(x.to_owned()));
    let yhat_test = test_dataset
        .features
        .map_axis(Axis(1), |x| decision_tree.predict(x.to_owned()));
    let prediction_duration = start_prediction.elapsed().unwrap().as_millis();
    println!("Prediction took {} ms", prediction_duration);

    //println!("yhat train shape: {:?}", yhat_train.shape());
    //println!("yhat test shape: {:?}", yhat_test.shape());

    let mae_train = (&yhat_train - &train_dataset.labels)
        .map(|y| y.abs())
        .mean()
        .unwrap();
    let mae_test = (&yhat_test - &test_dataset.labels)
        .map(|y| y.abs())
        .mean()
        .unwrap();

    println!("");

    let _train_accuracy = (1.0-mae_train)*100.0;
    let _test_accuracy = (1.0-mae_test)*100.0;

    //println!("Tree size: {}", &decision_tree.size());

    println!("Train Accuracy: {:?} %", _train_accuracy);

    println!("Test Accuracy: {:?} %", _test_accuracy);

    //println!("MAE Train: {:?}", mae_train);
    //println!("MAE Test: {:?}", mae_test);

    Ok(())
}