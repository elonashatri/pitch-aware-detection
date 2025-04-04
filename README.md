pitch-aware-detection/
├── config/
│   └── default.yaml            # Configuration parameters
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset loading code
│   └── transforms.py           # Data augmentation transforms
├── losses/
│   ├── __init__.py
│   ├── detection_loss.py       # Standard detection losses
│   ├── hierarchy_loss.py       # Combined hierarchical loss
│   ├── relationship_loss.py    # Relationship modeling losses
│   └── staff_loss.py           # Staffline-specific losses
├── models/
│   ├── __init__.py
│   ├── backbones.py            # Backbone network implementations
│   ├── element_detector.py     # Musical element detector
│   ├── omr_model.py            # Main model class
│   ├── relationship.py         # Relationship modeling module
│   ├── staffline_detector.py   # Specialized staffline detector
│   └── system_detector.py      # System and staff detection modules
├── utils/
│   ├── __init__.py
│   ├── anchors.py              # Anchor generation utilities
│   ├── evaluation.py           # Evaluation metrics
│   └── visualization.py        # Visualization tools
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
└── inference.py                # Inference script for new images



