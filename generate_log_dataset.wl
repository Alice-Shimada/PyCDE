#!/usr/bin/env wolfram

(*
Generate numerical derivative data for the PySR log-regression workflow.

The input .m file should return either:
1) a plain expression, e.g. 4/3 Log[1 - x] - 2/5 Log[1 - x - y] + 2/5 Log[y]
2) or an association, e.g.
   <|"Expression" -> expr, "Variables" -> {x, y}|>

Example:
/usr/local/Wolfram/Mathematica/14.3/Executables/wolfram -noprompt -script generate_log_dataset.wl \
  --function-file examples/3l4p1m_function.m \
  --output-dir /tmp/pysr_generated \
  --samples 200 \
  --working-precision 60
*)

ClearAll[
  ScriptArguments,
  ParseCLIArguments,
  InferVariables,
  LoadFunctionSpecification,
  RandomRationalValue,
  RandomRationalPoint,
  NumericRealVectorQ,
  EvaluateDerivativesAtPoint,
  PythonNumberString,
  WriteNumericColumn,
  DerivativeColumnName,
  BuildPaperTemplateCombineString,
  BuildUnaryOperators,
  BuildNestedConstraints,
  OperatorModeName,
  DefaultRegressorOptions,
  BuildConfigAssociation,
  GenerateLogDataset,
  PrintUsage,
  Main
];

ScriptArguments[] := Module[{scriptIndex},
  scriptIndex = FirstCase[
    MapIndexed[
      If[StringMatchQ[#1, ___ ~~ ".wl"] || StringMatchQ[#1, ___ ~~ ".m"], #2[[1]], Nothing] &,
      $CommandLine
    ],
    _Integer,
    Missing["NotFound"]
  ];
  If[MissingQ[scriptIndex], {}, Drop[$CommandLine, scriptIndex]]
];

ParseCLIArguments[args_List] := Module[{assoc = <||>, i = 1, key},
  While[i <= Length[args],
    key = args[[i]];
    If[! StringStartsQ[key, "--"],
      Print["ERROR: expected an option starting with --, got: ", key];
      Exit[1];
    ];
    key = StringDrop[key, 2];
    If[i == Length[args] || StringStartsQ[args[[i + 1]], "--"],
      assoc[key] = True;
      i += 1,
      assoc[key] = args[[i + 1]];
      i += 2
    ];
  ];
  assoc
];

InferVariables[expr_] := SortBy[
  DeleteDuplicates @ Cases[
    Unevaluated[expr],
    s_Symbol /; Context[Unevaluated[s]] =!= "System`" &&
      Unevaluated[s] =!= I && Unevaluated[s] =!= E && Unevaluated[s] =!= Pi,
    Infinity
  ],
  SymbolName
];

LoadFunctionSpecification[file_String, variableOverride_: Automatic] := Module[
  {raw, expr, vars},
  raw = Get[file];
  Which[
    AssociationQ[raw],
      expr = Lookup[raw, "Expression", Lookup[raw, "f", Missing["Expression"]]];
      vars = Replace[variableOverride, Automatic :> Lookup[raw, "Variables", Automatic]],
    True,
      expr = raw;
      vars = variableOverride
  ];

  If[MissingQ[expr] || expr === Null,
    Print["ERROR: function file must return either an expression or an association with key \"Expression\"."];
    Exit[1];
  ];

  If[vars === Automatic,
    vars = InferVariables[expr]
  ];

  If[! ListQ[vars] || vars === {} || ! VectorQ[vars, MatchQ[_Symbol]],
    Print["ERROR: could not determine the variable list. Provide \"Variables\" in the .m file."];
    Exit[1];
  ];

  <|"Expression" -> expr, "Variables" -> vars|>
];

RandomRationalValue[integerRange_Integer, denominatorMax_Integer] := Module[{num, den},
  den = RandomInteger[{1, denominatorMax}];
  num = RandomInteger[{-integerRange, integerRange}];
  Rationalize[num/den, 0]
];

RandomRationalPoint[vars_List, integerRange_Integer, denominatorMax_Integer] :=
  AssociationThread[vars, Table[RandomRationalValue[integerRange, denominatorMax], {Length[vars]}]];

NumericRealVectorQ[vals_List] :=
  VectorQ[vals, NumericQ] &&
  FreeQ[vals, Indeterminate | ComplexInfinity | DirectedInfinity[_]] &&
  AllTrue[vals, PossibleZeroQ[Im[#]] &];

EvaluateDerivativesAtPoint[derivatives_List, point_Association, workingPrecision_Integer, minPrecision_Integer] := Module[
  {values},
  values = Quiet @ Check[N[derivatives /. Normal[point], workingPrecision], $Failed];
  If[values === $Failed || ! ListQ[values] || ! NumericRealVectorQ[values],
    Return[$Failed]
  ];
  values = Re /@ values;
  If[AnyTrue[values, Precision[#] < minPrecision &],
    Return[$Failed]
  ];
  values
];

PythonNumberString[value_, workingPrecision_Integer] :=
  ToString[CForm[SetPrecision[N[value, workingPrecision], workingPrecision]]];

WriteNumericColumn[path_String, values_List, workingPrecision_Integer] := Module[{lines},
  lines = PythonNumberString[#, workingPrecision] & /@ values;
  Export[path, StringRiffle[lines, "\n"] <> "\n", "String"]
];

DerivativeColumnName[var_Symbol] := "d" <> SymbolName[Unevaluated[var]];

BuildPaperTemplateCombineString[varNames_List, derivativeNames_List] := Module[{argList},
  If[Length[varNames] =!= 2 || Length[derivativeNames] =!= 2,
    Print[
      "ERROR: the current paper-style template is defined only for two variables. Got variables: ",
      StringRiffle[varNames, ", "]
    ];
    Exit[1];
  ];
  argList = StringRiffle[varNames, ", "];
  StringRiffle[
    {
      "fdx = D(f, 1)(" <> argList <> ")",
      "fdy = D(f, 2)(" <> argList <> ")",
      "",
      "abs2(fdx - " <> derivativeNames[[1]] <> ") + abs2(fdy - " <> derivativeNames[[2]] <> ")"
    },
    "\n"
  ]
];

BuildUnaryOperators[allowSqrt_: False] :=
  If[TrueQ[allowSqrt], {"log", "sqrt"}, {"log"}];

BuildNestedConstraints[allowSqrt_: False] :=
  If[
    TrueQ[allowSqrt],
    <|
      "sqrt" -> <|"sqrt" -> 0, "log" -> 0|>,
      "log" -> <|"sqrt" -> 1, "log" -> 0|>
    |>,
    <|"log" -> <|"log" -> 0|>|>
  ];

OperatorModeName[allowSqrt_: False] :=
  If[TrueQ[allowSqrt], "log-sqrt", "log-only"];

DefaultRegressorOptions[allowSqrt_: False] := <|
  "procs" -> 40,
  "populations" -> 120,
  "niterations" -> 500,
  "population_size" -> 120,
  "binary_operators" -> {"+", "-", "*", "/"},
  "unary_operators" -> BuildUnaryOperators[allowSqrt],
  "model_selection" -> "best",
  "nested_constraints" -> BuildNestedConstraints[allowSqrt],
  "complexity_of_constants" -> 5,
  "elementwise_loss" -> "my_loss(predicted, target) = predicted",
  "maxsize" -> 100,
  "turbo" -> True,
  "bumper" -> True,
  "precision" -> 64
|>;

BuildConfigAssociation[runName_String, varNames_List, derivativeNames_List, allowSqrt_: False] := Module[{regressorOptions},
  regressorOptions = Join[
    DefaultRegressorOptions[allowSqrt],
    <|"variable_names" -> Join[varNames, derivativeNames]|>
  ];
  <|
  "run_name" -> runName,
  "data_dir" -> ".",
  "feature_columns" -> Join[
    Map[<|"name" -> #, "file" -> (# <> ".txt")|> &, varNames],
    Map[<|"name" -> #, "file" -> (# <> ".txt")|> &, derivativeNames]
  ],
  "target" -> <|"mode" -> "feature", "name" -> First[varNames]|>,
  "template" -> <|
    "expressions" -> {"f"},
    "variable_names" -> Join[varNames, derivativeNames],
    "combine" -> BuildPaperTemplateCombineString[varNames, derivativeNames]
  |>,
  "regressor_kwargs" -> regressorOptions,
  "analysis" -> <|
    "expression_variables" -> varNames,
    "derivative_columns" -> MapThread[<|"variable" -> #1, "column" -> #2|> &, {varNames, derivativeNames}],
    "exact_loss_threshold" -> 1.*^-20,
    "max_abs_error_threshold" -> 1.*^-10,
    "rmse_threshold" -> 1.*^-10,
    "rational_tolerance" -> 1.*^-10
  |>,
  "notes" -> {
    "Auto-generated by generate_log_dataset.wl.",
    "The fit target is a dummy feature column because the template defines the true multi-variable loss.",
    "The source values were sampled at rational kinematic points and exported with high decimal precision.",
    "Operator mode: " <> OperatorModeName[allowSqrt]
  }
|>
];

Options[GenerateLogDataset] = {
  "FunctionFile" -> Missing["NotSpecified"],
  "OutputDir" -> Missing["NotSpecified"],
  "Variables" -> Automatic,
  "Samples" -> 200,
  "WorkingPrecision" -> 60,
  "MinPrecision" -> 30,
  "IntegerRange" -> 9,
  "DenominatorMax" -> 13,
  "RandomSeed" -> Automatic,
  "AllowSqrt" -> False
};

GenerateLogDataset[OptionsPattern[]] := Module[
  {
    functionFile, outputDir, variableOverride, sampleCount, workingPrecision,
    minPrecision, integerRange, denominatorMax, randomSeed, allowSqrt,
    spec, expr, vars, varNames, derivativeNames, derivatives,
    points = {}, derivativeRows = {}, attempts = 0, maxAttempts, point, values,
    manifest, config, i
  },
  functionFile = ExpandFileName[OptionValue["FunctionFile"]];
  outputDir = ExpandFileName[OptionValue["OutputDir"]];
  variableOverride = OptionValue["Variables"];
  sampleCount = OptionValue["Samples"];
  workingPrecision = OptionValue["WorkingPrecision"];
  minPrecision = OptionValue["MinPrecision"];
  integerRange = OptionValue["IntegerRange"];
  denominatorMax = OptionValue["DenominatorMax"];
  randomSeed = OptionValue["RandomSeed"];
  allowSqrt = TrueQ[OptionValue["AllowSqrt"]];

  If[! FileExistsQ[functionFile],
    Print["ERROR: function file does not exist: ", functionFile];
    Exit[1];
  ];

  If[randomSeed =!= Automatic,
    SeedRandom[randomSeed]
  ];

  spec = LoadFunctionSpecification[functionFile, variableOverride];
  expr = spec["Expression"];
  vars = spec["Variables"];
  varNames = SymbolName /@ vars;
  derivativeNames = DerivativeColumnName /@ vars;
  derivatives = D[expr, #] & /@ vars;

  maxAttempts = 1000 * sampleCount;
  While[Length[points] < sampleCount && attempts < maxAttempts,
    attempts += 1;
    point = RandomRationalPoint[vars, integerRange, denominatorMax];
    values = EvaluateDerivativesAtPoint[derivatives, point, workingPrecision, minPrecision];
    If[values =!= $Failed,
      AppendTo[points, point];
      AppendTo[derivativeRows, values];
    ];
  ];

  If[Length[points] < sampleCount,
    Print[
      "ERROR: only generated ", Length[points], " valid samples after ", attempts,
      " attempts. Try a larger IntegerRange or DenominatorMax, or adjust the function domain."
    ];
    Exit[1];
  ];

  CreateDirectory[outputDir, CreateIntermediateDirectories -> True];

  Do[
    WriteNumericColumn[
      FileNameJoin[{outputDir, varNames[[i]] <> ".txt"}],
      Lookup[points, vars[[i]]],
      workingPrecision
    ],
    {i, Length[vars]}
  ];

  Do[
    WriteNumericColumn[
      FileNameJoin[{outputDir, derivativeNames[[i]] <> ".txt"}],
      derivativeRows[[All, i]],
      workingPrecision
    ],
    {i, Length[vars]}
  ];

  config = BuildConfigAssociation[FileBaseName[functionFile], varNames, derivativeNames, allowSqrt];
  Export[FileNameJoin[{outputDir, "generated_config.json"}], config, "RawJSON"];

  CopyFile[functionFile, FileNameJoin[{outputDir, FileNameTake[functionFile]}], OverwriteTarget -> True];

  manifest = <|
    "function_file" -> functionFile,
    "expression" -> ToString[InputForm[expr]],
    "variables" -> varNames,
    "derivative_columns" -> derivativeNames,
    "derivatives" -> AssociationThread[varNames, ToString[InputForm[#]] & /@ derivatives],
    "operator_mode" -> OperatorModeName[allowSqrt],
    "allow_sqrt" -> allowSqrt,
    "unary_operators" -> BuildUnaryOperators[allowSqrt],
    "nested_constraints" -> BuildNestedConstraints[allowSqrt],
    "sample_count" -> sampleCount,
    "working_precision" -> workingPrecision,
    "min_precision" -> minPrecision,
    "integer_range" -> integerRange,
    "denominator_max" -> denominatorMax,
    "attempts" -> attempts,
    "output_dir" -> outputDir
  |>;
  Export[FileNameJoin[{outputDir, "dataset_manifest.json"}], manifest, "RawJSON"];

  WriteString[$Output, "Generated dataset in: " <> outputDir <> "\n"];
  WriteString[$Output, "Config file: " <> FileNameJoin[{outputDir, "generated_config.json"}] <> "\n"];
  WriteString[$Output, "Variables: " <> StringRiffle[varNames, ", "] <> "\n"];
  WriteString[$Output, "Operator mode: " <> OperatorModeName[allowSqrt] <> "\n"];
];

PrintUsage[] := WriteString[
  $Output,
  StringRiffle[
    {
      "Usage:",
      "  wolfram -noprompt -script generate_log_dataset.wl --function-file path/to/f.m --output-dir path [options]",
      "",
      "Options:",
      "  --samples N                Number of sample points (default: 200)",
      "  --working-precision N      Decimal precision used for exported values (default: 60)",
      "  --min-precision N          Minimum accepted precision for derivative values (default: 30)",
      "  --integer-range N          Numerators sampled from [-N, N] (default: 9)",
      "  --denominator-max N        Denominators sampled from [1, N] (default: 13)",
      "  --random-seed N            Optional deterministic seed",
      "  --allow-sqrt              Switch to the paper's log+sqrt operator mode",
      "                           (unary_operators=[log,sqrt], with nested log/sqrt constraints)",
      "",
      "The .m file should return either a plain expression or an association like:",
      "  <|\"Expression\" -> expr, \"Variables\" -> {x, y}|>"
    },
    "\n"
  ] <> "\n"
];

Main[] := Module[{args, functionFile, outputDir, samples, workingPrecision, minPrecision, integerRange, denominatorMax, randomSeed, allowSqrt},
  args = ParseCLIArguments[ScriptArguments[]];
  If[args === <||> || KeyExistsQ[args, "help"],
    PrintUsage[];
    Exit[0];
  ];

  functionFile = Lookup[args, "function-file", Missing["NotSpecified"]];
  outputDir = Lookup[args, "output-dir", Missing["NotSpecified"]];

  If[MissingQ[functionFile] || MissingQ[outputDir],
    Print["ERROR: both --function-file and --output-dir are required."];
    PrintUsage[];
    Exit[1];
  ];

  samples = ToExpression[Lookup[args, "samples", "200"]];
  workingPrecision = ToExpression[Lookup[args, "working-precision", "60"]];
  minPrecision = ToExpression[Lookup[args, "min-precision", "30"]];
  integerRange = ToExpression[Lookup[args, "integer-range", "9"]];
  denominatorMax = ToExpression[Lookup[args, "denominator-max", "13"]];
  randomSeed = If[KeyExistsQ[args, "random-seed"], ToExpression[args["random-seed"]], Automatic];
  allowSqrt = TrueQ[Lookup[args, "allow-sqrt", False]];

  GenerateLogDataset[
    "FunctionFile" -> functionFile,
    "OutputDir" -> outputDir,
    "Samples" -> samples,
    "WorkingPrecision" -> workingPrecision,
    "MinPrecision" -> minPrecision,
    "IntegerRange" -> integerRange,
    "DenominatorMax" -> denominatorMax,
    "RandomSeed" -> randomSeed,
    "AllowSqrt" -> allowSqrt
  ];
];

If[$FrontEnd === Null, Main[]];
