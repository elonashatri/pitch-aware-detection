import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re

class MusicNotationAnalyzer:
    def __init__(self, data_dir):
        """
        Initialize the analyzer with the directory containing XML data.
        
        Args:
            data_dir: Path to directory containing XML files
        """
        self.data_dir = Path(data_dir)
        self.staffline_stats = defaultdict(list)
        self.notehead_stats = defaultdict(list)
        self.staff_spacings = []
        self.pitch_distributions = Counter()
        self.line_pos_to_pitch = defaultdict(list)
        self.staff_completeness = []
        self.clef_distribution = Counter()
        self.system_stats = defaultdict(list)
        
    def analyze_xml_file(self, xml_file):
        """Parse and analyze a single XML file"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get all nodes
            nodes = root.findall('.//Node')
            
            # Group nodes by staff_id and spacing_run_id
            staff_groups = defaultdict(list)
            system_groups = defaultdict(list)
            
            for node in nodes:
                class_name = node.find('ClassName').text if node.find('ClassName') is not None else ""
                
                # Extract common node properties
                node_data = {
                    'id': node.find('Id').text if node.find('Id') is not None else "",
                    'class_name': class_name,
                    'top': int(node.find('Top').text) if node.find('Top') is not None else 0,
                    'left': int(node.find('Left').text) if node.find('Left') is not None else 0,
                    'width': int(node.find('Width').text) if node.find('Width') is not None else 0,
                    'height': int(node.find('Height').text) if node.find('Height') is not None else 0,
                }
                
                # Extract dataItems
                data_items = node.findall('.//DataItem')
                for item in data_items:
                    key = item.get('key')
                    value = item.text
                    value_type = item.get('type')
                    
                    # Convert value based on type
                    if value_type == 'int':
                        value = int(value)
                    elif value_type == 'float':
                        value = float(value)
                        
                    node_data[key] = value
                
                # Group by staff_id if available
                if 'staff_id' in node_data:
                    staff_groups[node_data['staff_id']].append(node_data)
                
                # Group by spacing_run_id if available
                if 'spacing_run_id' in node_data:
                    system_groups[node_data['spacing_run_id']].append(node_data)
                
                # Analyze stafflines
                if class_name == 'kStaffLine':
                    self.staffline_stats['width'].append(node_data['width'])
                    self.staffline_stats['height'].append(node_data['height'])
                    self.staffline_stats['top'].append(node_data['top'])
                    self.staffline_stats['left'].append(node_data['left'])
                
                # Analyze noteheads
                if 'notehead' in class_name.lower():
                    self.notehead_stats['width'].append(node_data['width'])
                    self.notehead_stats['height'].append(node_data['height'])
                    
                    # Analyze pitch information if available
                    if 'midi_pitch_code' in node_data and 'line_pos' in node_data:
                        pitch_code = node_data['midi_pitch_code']
                        line_pos = node_data['line_pos']
                        self.pitch_distributions[pitch_code] += 1
                        
                        # Map line_pos to midi_pitch_code
                        self.line_pos_to_pitch[line_pos].append(pitch_code)
                
                # Analyze clefs
                if class_name in ['gClef', 'fClef', 'cClef']:
                    self.clef_distribution[class_name] += 1
            
            # Analyze staff completeness (should have 5 stafflines)
            for staff_id, nodes in staff_groups.items():
                stafflines = [n for n in nodes if n['class_name'] == 'kStaffLine']
                self.staff_completeness.append(len(stafflines))
                
                # If we have at least 2 stafflines, calculate spacing
                if len(stafflines) >= 2:
                    tops = sorted([s['top'] for s in stafflines])
                    for i in range(len(tops) - 1):
                        self.staff_spacings.append(tops[i+1] - tops[i])
            
            # Analyze system properties
            for system_id, nodes in system_groups.items():
                stafflines = [n for n in nodes if n['class_name'] == 'kStaffLine']
                if stafflines:
                    self.system_stats['num_stafflines'].append(len(stafflines))
                    self.system_stats['width'].append(np.mean([s['width'] for s in stafflines]))
                    
                    # Calculate vertical range of system
                    tops = [s['top'] for s in stafflines]
                    self.system_stats['height'].append(max(tops) - min(tops) + 3)  # Add 3 for staffline height
                
            return True
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return False
    
    def analyze_all_files(self):
        """Analyze all XML files in the data directory"""
        xml_files = list(self.data_dir.glob('**/*.xml'))
        print(f"Found {len(xml_files)} XML files to analyze")
        
        successful = 0
        for i, file in enumerate(xml_files):
            if i % 10 == 0:
                print(f"Processing file {i+1}/{len(xml_files)}: {file.name}")
                
            if self.analyze_xml_file(file):
                successful += 1
        
        print(f"Successfully analyzed {successful}/{len(xml_files)} files")
    
    def generate_staffline_stats(self):
        """Generate statistics for stafflines"""
        if not self.staffline_stats['width']:
            return None
            
        stats = {
            'width': {
                'mean': np.mean(self.staffline_stats['width']),
                'median': np.median(self.staffline_stats['width']),
                'std': np.std(self.staffline_stats['width']),
                'min': np.min(self.staffline_stats['width']),
                'max': np.max(self.staffline_stats['width']),
            },
            'height': {
                'mean': np.mean(self.staffline_stats['height']),
                'median': np.median(self.staffline_stats['height']),
                'std': np.std(self.staffline_stats['height']),
                'min': np.min(self.staffline_stats['height']),
                'max': np.max(self.staffline_stats['height']),
            },
            'aspect_ratio': {
                'mean': np.mean([w/h for w, h in zip(self.staffline_stats['width'], self.staffline_stats['height'])]),
                'median': np.median([w/h for w, h in zip(self.staffline_stats['width'], self.staffline_stats['height'])]),
                'min': np.min([w/h for w, h in zip(self.staffline_stats['width'], self.staffline_stats['height'])]),
                'max': np.max([w/h for w, h in zip(self.staffline_stats['width'], self.staffline_stats['height'])]),
            }
        }
        
        return stats
    
    def generate_staff_spacing_stats(self):
        """Generate statistics for staff line spacing"""
        if not self.staff_spacings:
            return None
            
        stats = {
            'mean': np.mean(self.staff_spacings),
            'median': np.median(self.staff_spacings),
            'std': np.std(self.staff_spacings),
            'min': np.min(self.staff_spacings),
            'max': np.max(self.staff_spacings),
        }
        
        return stats
    
    def generate_pitch_mapping(self):
        """Generate mapping from line positions to pitches"""
        mapping = {}
        
        for line_pos, pitches in self.line_pos_to_pitch.items():
            # Find most common pitch for this line position
            if pitches:
                most_common = Counter(pitches).most_common(1)[0][0]
                mapping[line_pos] = most_common
        
        return mapping
    
    def generate_staff_completeness_stats(self):
        """Generate statistics for staff completeness"""
        if not self.staff_completeness:
            return None
            
        counts = Counter(self.staff_completeness)
        stats = {
            'mean': np.mean(self.staff_completeness),
            'median': np.median(self.staff_completeness),
            'counts': {k: v for k, v in sorted(counts.items())}
        }
        
        return stats
    
    def plot_staffline_dimensions(self):
        """Create histogram of staffline dimensions"""
        if not self.staffline_stats['width']:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Width histogram
        sns.histplot(self.staffline_stats['width'], bins=30, ax=ax1)
        ax1.set_title('Staffline Width Distribution')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Count')
        
        # Height histogram
        sns.histplot(self.staffline_stats['height'], bins=30, ax=ax2)
        ax2.set_title('Staffline Height Distribution')
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('staffline_dimensions.png')
        plt.close()
    
    def plot_staff_spacing(self):
        """Create histogram of staffline spacing"""
        if not self.staff_spacings:
            return
            
        plt.figure(figsize=(10, 6))
        sns.histplot(self.staff_spacings, bins=30)
        plt.title('Staff Line Spacing Distribution')
        plt.xlabel('Spacing (pixels)')
        plt.ylabel('Count')
        plt.savefig('staff_spacing.png')
        plt.close()
    
    def plot_pitch_distribution(self):
        """Create histogram of pitches"""
        if not self.pitch_distributions:
            return
            
        plt.figure(figsize=(12, 6))
        pitches = sorted(self.pitch_distributions.keys())
        counts = [self.pitch_distributions[p] for p in pitches]
        
        # Create readable pitch labels
        pitch_labels = []
        for p in pitches:
            octave = p // 12 - 1  # MIDI octave conversion
            note_idx = p % 12
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            pitch_labels.append(f"{note_names[note_idx]}{octave}")
        
        plt.bar(pitch_labels, counts)
        plt.title('Pitch Distribution')
        plt.xlabel('Pitch')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('pitch_distribution.png')
        plt.close()
    
    def plot_staff_completeness(self):
        """Create histogram of staff completeness"""
        if not self.staff_completeness:
            return
            
        plt.figure(figsize=(10, 6))
        counts = Counter(self.staff_completeness)
        x = sorted(counts.keys())
        y = [counts[k] for k in x]
        
        plt.bar(x, y)
        plt.title('Staff Completeness Distribution')
        plt.xlabel('Number of Stafflines per Staff')
        plt.ylabel('Count')
        plt.xticks(x)
        plt.savefig('staff_completeness.png')
        plt.close()
    
    def plot_clef_distribution(self):
        """Create bar chart of clef distribution"""
        if not self.clef_distribution:
            return
            
        plt.figure(figsize=(8, 6))
        clefs = list(self.clef_distribution.keys())
        counts = [self.clef_distribution[c] for c in clefs]
        
        plt.bar(clefs, counts)
        plt.title('Clef Distribution')
        plt.xlabel('Clef Type')
        plt.ylabel('Count')
        plt.savefig('clef_distribution.png')
        plt.close()
    
    def plot_line_pos_to_pitch(self):
        """Visualize line position to pitch mapping"""
        if not self.line_pos_to_pitch:
            return
        
        # Sort line positions in a sensible order
        def line_pos_sort_key(pos):
            # Extract line number
            match = re.match(r"([LS])(-?\d+)", pos)
            if not match:
                return (2, 0)  # Default
                
            prefix, num = match.groups()
            # S prefix comes before L
            prefix_val = 0 if prefix == 'S' else 1
            return (prefix_val, int(num))
        
        line_positions = sorted(self.line_pos_to_pitch.keys(), key=line_pos_sort_key)
        
        plt.figure(figsize=(12, 8))
        
        for i, pos in enumerate(line_positions):
            if not self.line_pos_to_pitch[pos]:
                continue
                
            # Calculate mean pitch for this position
            mean_pitch = np.mean(self.line_pos_to_pitch[pos])
            
            # Plot distribution
            y = [i] * len(self.line_pos_to_pitch[pos])
            plt.scatter(self.line_pos_to_pitch[pos], y, alpha=0.2, s=5)
            
            # Plot mean
            plt.axvline(x=mean_pitch, ymin=(i-0.2)/len(line_positions), ymax=(i+0.2)/len(line_positions), 
                       color='red', linewidth=2)
        
        plt.yticks(range(len(line_positions)), line_positions)
        plt.title('Line Position to Pitch Mapping')
        plt.xlabel('MIDI Pitch')
        plt.ylabel('Line Position')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('line_pos_to_pitch.png')
        plt.close()
    

    def generate_statistics_report(self):
        """Generate comprehensive statistics for the dataset"""
        
        staffline_stats = self.generate_staffline_stats()
        staff_spacing_stats = self.generate_staff_spacing_stats()
        pitch_mapping = self.generate_pitch_mapping()
        staff_completeness_stats = self.generate_staff_completeness_stats()
        
        # Convert NumPy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            return obj
        
        report = {
            "staffline_statistics": convert_to_native(staffline_stats),
            "staff_spacing_statistics": convert_to_native(staff_spacing_stats),
            "staff_completeness": convert_to_native(staff_completeness_stats),
            "clef_distribution": convert_to_native(dict(self.clef_distribution)),
            "line_position_to_pitch_mapping": convert_to_native(pitch_mapping),
            "system_statistics": convert_to_native({
                "num_stafflines": {
                    "mean": np.mean(self.system_stats['num_stafflines']) if self.system_stats['num_stafflines'] else 0,
                    "median": np.median(self.system_stats['num_stafflines']) if self.system_stats['num_stafflines'] else 0,
                },
                "width": {
                    "mean": np.mean(self.system_stats['width']) if self.system_stats['width'] else 0,
                    "median": np.median(self.system_stats['width']) if self.system_stats['width'] else 0,
                },
                "height": {
                    "mean": np.mean(self.system_stats['height']) if self.system_stats['height'] else 0,
                    "median": np.median(self.system_stats['height']) if self.system_stats['height'] else 0,
                }
            })
        }
        
        # Save report as JSON
        with open('music_notation_statistics.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self.plot_staffline_dimensions()
        self.plot_staff_spacing()
        self.plot_pitch_distribution()
        self.plot_staff_completeness()
        self.plot_clef_distribution()
        self.plot_line_pos_to_pitch()
        
        return report

def analyze_dataset(data_dir):
    """Run complete analysis on the dataset"""
    analyzer = MusicNotationAnalyzer(data_dir)
    analyzer.analyze_all_files()
    report = analyzer.generate_statistics_report()
    
    print("\n===== MUSIC NOTATION DATASET STATISTICS =====\n")
    
    # Print key statistics
    if report["staffline_statistics"]:
        print(f"Staffline Width: mean={report['staffline_statistics']['width']['mean']:.2f}, "
              f"median={report['staffline_statistics']['width']['median']}, "
              f"min={report['staffline_statistics']['width']['min']}, "
              f"max={report['staffline_statistics']['width']['max']}")
        
        print(f"Staffline Height: mean={report['staffline_statistics']['height']['mean']:.2f}, "
              f"median={report['staffline_statistics']['height']['median']}, "
              f"min={report['staffline_statistics']['height']['min']}, "
              f"max={report['staffline_statistics']['height']['max']}")
    
    if report["staff_spacing_statistics"]:
        print(f"\nStaff Spacing: mean={report['staff_spacing_statistics']['mean']:.2f}, "
              f"median={report['staff_spacing_statistics']['median']}, "
              f"min={report['staff_spacing_statistics']['min']}, "
              f"max={report['staff_spacing_statistics']['max']}")
    
    if report["staff_completeness"]:
        print(f"\nStaff Completeness: mean={report['staff_completeness']['mean']:.2f}, "
              f"median={report['staff_completeness']['median']}")
        print(f"Staff Line Counts: {report['staff_completeness']['counts']}")
    
    print(f"\nClef Distribution: {report['clef_distribution']}")
    
    print("\nLine Position to Pitch Mapping (partial):")
    for pos, pitch in list(report["line_position_to_pitch_mapping"].items())[:10]:
        print(f"  {pos}: {pitch}")
    
    print("\nVisualization files saved to current directory.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze music notation XML dataset")
    parser.add_argument("data_dir", help="Directory containing XML files")
    
    args = parser.parse_args()
    analyze_dataset(args.data_dir)