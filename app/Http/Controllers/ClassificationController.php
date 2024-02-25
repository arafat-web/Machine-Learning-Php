<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Persisters\Filesystem;

class ClassificationController extends Controller
{
    public function train()
    {
        $samples = [
            ['The team played an amazing game of soccer'],
            ['The new programming language has been released'],
            ['The match between the two teams was incredible'],
            ['The new tech gadget has been launched'],
        ];

        $labels = [
            'sports',
            'technology',
            'sports',
            'technology',
        ];

        $dataset = new Labeled($samples, $labels);

        $estimator = new Pipeline([
            new WordCountVectorizer(10000, 1, 1, new Word()),
            new TfIdfTransformer(),
        ], new KNearestNeighbors(4));

        $estimator->train($dataset);

        $this->saveModel($estimator);

        echo "Training completed and model saved.\n";
    }

    private function saveModel($estimator)
    {
        $persister = new Filesystem(storage_path('app/model.rbx'));
        $model = new PersistentModel($estimator, $persister);
        $model->save();
    }

    public function predictNewSamples(Request $request)
    {
        $samples = [
            ['The team played an amazing game of soccer, showing excellent teamwork and strategy.'],
            ['The latest programming language release introduces features that enhance coding efficiency.'],
            ['An incredible match between two top teams ended in a thrilling draw last night.'],
            ['This new tech gadget includes features never before seen, setting a new standard in the industry.'],
        ];
        $persister = new Filesystem(storage_path('app/model.rbx'));
        $model = PersistentModel::load($persister);
        $newSamples = $samples;

        $dataset = new Unlabeled($newSamples);
        $predictions = $model->predict($dataset);

        return response()->json([
            'predictions' => $predictions,
        ]);
    }
}
