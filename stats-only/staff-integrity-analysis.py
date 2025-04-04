import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime

class StaffIntegrityChecker:
    """
    Analyzes stafflines for integrity and completeness to help build 
    robust staffline detection models.
    """
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.staff_info = []
        self.system_info = []
        self.line_naming_patterns = Counter()
        self.staffline_completeness = []
        self.incomplete_staves = []
        self.staff_spacings_by_system = defaultdict(list)
        self.staff_dimensions = []
    
    def process_all_files(self):
        """Process all XML files in the data directory"""
        xml_files = list(self.data_dir.glob('**/*.xml'))
        print(f"Found {len(xml_files)} XML files")
        
        success_count = 0
        for file in tqdm(xml_files, desc="Processing XML files"):
            if self.process_file(file):
                success_count += 1
        
        print(f"Successfully processed {success_count}/{len(xml_files)} files")
    
    def process_file(self, xml_file):
        """Process an individual XML file"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Check if this is our expected XML format
            page_nodes = root.findall('.//Page')
            if not page_nodes:
                print(f"Skipping {xml_file} - unexpected format")
                return False
                
            for page in page_nodes:
                # Group nodes by staff_id and system
                staffs_by_id = defaultdict(list)
                systems_by_id = defaultdict(list)
                nodes = page.findall('.//Node')
                
                for node in nodes:
                    # Basic node properties
                    class_name = node.find('ClassName')
                    class_name = class_name.text if class_name is not None else ""
                    
                    # Only process nodes with class names
                    if not class_name:
                        continue
                        
                    node_info = {
                        'class_name': class_name,
                        'id': node.find('Id').text if node.find('Id') is not None else "",
                        'top': int(node.find('Top').text) if node.find('Top') is not None else 0,
                        'left': int(node.find('Left').text) if node.find('Left') is not None else 0,
                        'width': int(node.find('Width').text) if node.find('Width') is not None else 0,
                        'height': int(node.find('Height').text) if node.find('Height') is not None else 0,
                    }
                    
                    # Extract name attribute if available
                    name_attr = node.get('name')
                    if name_attr:
                        node_info['name'] = name_attr
                        
                        # Track staffline naming patterns
                        if class_name == 'kStaffLine':
                            self.line_naming_patterns[name_attr] += 1
                    
                    # Extract data items
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
                            
                        node_info[key] = value
                    
                    # Group by staff and system
                    if 'ordered_staff_id' in node_info:
                        staffs_by_id[node_info['ordered_staff_id']].append(node_info)
                    
                    if 'spacing_run_id' in node_info:
                        systems_by_id[node_info['spacing_run_id']].append(node_info)
                
                # Analyze staff completeness
                for staff_id, nodes in staffs_by_id.items():
                    stafflines = [n for n in nodes if n['class_name'] == 'kStaffLine']
                    
                    # Get the system ID for this staff if available
                    system_id = stafflines[0].get('spacing_run_id') if stafflines else None
                    
                    staff_record = {
                        'file': xml_file.name,
                        'ordered_staff_id': staff_id,
                        'system_id': system_id,
                        'num_stafflines': len(stafflines),
                        'staffline_names': [s.get('name', '') for s in stafflines],
                        'complete': len(stafflines) == 5
                    }
                    
                    self.staff_info.append(staff_record)
                    
                    if not staff_record['complete']:
                        self.incomplete_staves.append(staff_record)
                    
                    # Check staff line vertical ordering and spacing
                    if len(stafflines) >= 2:
                        # Sort by vertical position
                        stafflines_sorted = sorted(stafflines, key=lambda x: x['top'])
                        
                        # Calculate spacings
                        spacings = []
                        for i in range(len(stafflines_sorted) - 1):
                            spacing = stafflines_sorted[i+1]['top'] - stafflines_sorted[i]['top']
                            spacings.append(spacing)
                            
                            if system_id:
                                self.staff_spacings_by_system[system_id].append(spacing)
                        
                        # Save staff dimensions
                        top = min(s['top'] for s in stafflines)
                        bottom = max(s['top'] + s['height'] for s in stafflines)
                        left = min(s['left'] for s in stafflines)
                        right = max(s['left'] + s['width'] for s in stafflines)
                        
                        self.staff_dimensions.append({
                            'ordered_staff_id': staff_id,
                            'system_id': system_id,
                            'top': top,
                            'bottom': bottom,
                            'left': left,
                            'right': right,
                            'height': bottom - top,
                            'width': right - left,
                            'mean_spacing': np.mean(spacings) if spacings else 0,
                            'std_spacing': np.std(spacings) if spacings else 0,
                            'min_spacing': min(spacings) if spacings else 0,
                            'max_spacing': max(spacings) if spacings else 0,
                        })
                
                # Analyze system properties
                for system_id, nodes in systems_by_id.items():
                    stafflines = [n for n in nodes if n['class_name'] == 'kStaffLine']
                    staves = set(n['ordered_staff_id'] for n in stafflines if 'ordered_staff_id' in n)
                    
                    if stafflines:
                        self.system_info.append({
                            'system_id': system_id,
                            'num_stafflines': len(stafflines),
                            'num_staves': len(staves),
                            'mean_width': np.mean([s['width'] for s in stafflines]),
                            'mean_height': np.mean([s['height'] for s in stafflines]),
                        })
            
            return True
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return False
    
    def analyze_staff_completeness(self):
        """Analyze staff completeness statistics"""
        if not self.staff_info:
            return
        
        staff_df = pd.DataFrame(self.staff_info)
        
        complete_counts = staff_df['complete'].value_counts()
        staffline_counts = staff_df['num_stafflines'].value_counts().sort_index()
        
        print("\n===== STAFF COMPLETENESS ANALYSIS =====")
        print(f"Total staves: {len(staff_df)}")
        print(f"Complete staves (5 stafflines): {complete_counts.get(True, 0)} ({complete_counts.get(True, 0)/len(staff_df)*100:.1f}%)")
        print(f"Incomplete staves: {complete_counts.get(False, 0)} ({complete_counts.get(False, 0)/len(staff_df)*100:.1f}%)")
        print("\nStaffline count distribution:")
        for count, frequency in staffline_counts.items():
            print(f"  {count} stafflines: {frequency} staves ({frequency/len(staff_df)*100:.1f}%)")
        
        # Plot staffline count distribution
        plt.figure(figsize=(10, 6))
        ax = staffline_counts.plot(kind='bar')
        plt.title('Distribution of Stafflines per Staff')
        plt.xlabel('Number of Stafflines')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('staffline_count_distribution.png')
        plt.close()
        
        return {
            'total_staves': len(staff_df),
            'complete_staves': int(complete_counts.get(True, 0)),
            'incomplete_staves': int(complete_counts.get(False, 0)),
            'staffline_counts': staffline_counts.to_dict()
        }
    
    def analyze_staffline_naming(self):
        """Analyze staffline naming patterns"""
        if not self.line_naming_patterns:
            return
        
        print("\n===== STAFFLINE NAMING PATTERNS =====")
        total = sum(self.line_naming_patterns.values())
        
        print(f"Staffline naming patterns (total: {total}):")
        for name, count in self.line_naming_patterns.most_common():
            print(f"  {name}: {count} ({count/total*100:.1f}%)")
        
        return dict(self.line_naming_patterns)
    
    def analyze_staff_spacing(self):
        """Analyze staff spacing consistency"""
        if not self.staff_dimensions:
            return
        
        staff_df = pd.DataFrame(self.staff_dimensions)
        
        # Check for system-specific spacing patterns
        if self.staff_spacings_by_system:
            print("\n===== SYSTEM-SPECIFIC SPACING ANALYSIS =====")
            mean_spacings = {}
            
            for system_id, spacings in self.staff_spacings_by_system.items():
                mean_spacings[system_id] = np.mean(spacings)
            
            # Calculate variance in spacing across systems
            system_spacing_mean = np.mean(list(mean_spacings.values()))
            system_spacing_std = np.std(list(mean_spacings.values()))
            
            print(f"Mean system spacing: {system_spacing_mean:.2f} pixels")
            print(f"System spacing std: {system_spacing_std:.2f} pixels")
            print(f"System spacing variability: {system_spacing_std/system_spacing_mean*100:.2f}%")
            print(f"Number of systems analyzed: {len(mean_spacings)}")
            
            # Plot system spacing distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(list(mean_spacings.values()), bins=30)
            plt.title('Distribution of Mean Spacing Across Systems')
            plt.xlabel('Mean System Staffline Spacing (pixels)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig('system_spacing_distribution.png')
            plt.close()
        
        print("\n===== STAFF SPACING ANALYSIS =====")
        print(f"Mean staff height: {staff_df['height'].mean():.2f} pixels")
        print(f"Mean staff width: {staff_df['width'].mean():.2f} pixels")
        print(f"Mean staffline spacing: {staff_df['mean_spacing'].mean():.2f} pixels")
        print(f"Min staffline spacing: {staff_df['min_spacing'].min():.2f} pixels")
        print(f"Max staffline spacing: {staff_df['max_spacing'].max():.2f} pixels")
        print(f"Mean spacing std deviation within staff: {staff_df['std_spacing'].mean():.2f} pixels")
        
        # Plot staff spacing distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=staff_df, x='mean_spacing', bins=30)
        plt.title('Distribution of Mean Staffline Spacing')
        plt.xlabel('Mean Spacing (pixels)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('staffline_spacing_distribution.png')
        plt.close()
        
        # Plot staff dimension relationships
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=staff_df, x='width', y='height')
        plt.title('Staff Dimensions')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.tight_layout()
        plt.savefig('staff_dimensions.png')
        plt.close()
        
        # Create additional plots for pitch model design
        
        # Spacing consistency within staff
        plt.figure(figsize=(10, 6))
        sns.histplot(data=staff_df, x='std_spacing', bins=30)
        plt.title('Consistency of Staffline Spacing (Std Deviation Within Staff)')
        plt.xlabel('Standard Deviation of Spacing (pixels)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('staffline_spacing_consistency.png')
        plt.close()
        
        # Return detailed statistics for staff spacing
        return {
            'mean_height': float(staff_df['height'].mean()),
            'mean_width': float(staff_df['width'].mean()),
            'mean_spacing': float(staff_df['mean_spacing'].mean()),
            'min_spacing': float(staff_df['min_spacing'].min()),
            'max_spacing': float(staff_df['max_spacing'].max()),
            'mean_spacing_std': float(staff_df['std_spacing'].mean()),
            'spacing_percentiles': {
                '5%': float(staff_df['mean_spacing'].quantile(0.05)),
                '25%': float(staff_df['mean_spacing'].quantile(0.25)),
                '50%': float(staff_df['mean_spacing'].quantile(0.50)),
                '75%': float(staff_df['mean_spacing'].quantile(0.75)),
                '95%': float(staff_df['mean_spacing'].quantile(0.95)),
            }
        }
        
        

    def analyze_pitch_line_positions(self):
        """
        Analyze relationship between stafflines and pitch positions.
        This helps establish the mapping between line positions and pitches.
        """
        pitch_data = []
        
        # Process each staff to extract notehead positions relative to stafflines
        for xml_file in tqdm(list(self.data_dir.glob('**/*.xml')), desc="Analyzing pitch positions"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Group elements by staff_id
                staffs = defaultdict(list)
                nodes = root.findall('.//Node')
                
                for node in nodes:
                    class_name = node.find('ClassName')
                    class_name = class_name.text if class_name is not None else ""
                    
                    # Get data items including staff_id and pitch info
                    data_items = {}
                    for item in node.findall('.//DataItem'):
                        key = item.get('key')
                        value = item.text
                        value_type = item.get('type')
                        
                        if value_type == 'int':
                            value = int(value)
                        elif value_type == 'float':
                            value = float(value)
                            
                        data_items[key] = value
                    
                    if 'ordered_staff_id' in data_items:
                        # Save basic info
                        node_info = {
                            'class_name': class_name,
                            'top': int(node.find('Top').text) if node.find('Top') is not None else 0,
                            'left': int(node.find('Left').text) if node.find('Left') is not None else 0,
                            'width': int(node.find('Width').text) if node.find('Width') is not None else 0,
                            'height': int(node.find('Height').text) if node.find('Height') is not None else 0,
                            **data_items
                        }
                        
                        staffs[data_items['ordered_staff_id']].append(node_info)
                
                # For each staff, analyze noteheads relative to stafflines
                for staff_id, elements in staffs.items():
                    stafflines = [e for e in elements if e['class_name'] == 'kStaffLine']
                    noteheads = [e for e in elements if 'notehead' in e['class_name'].lower()]
                    
                    # Skip if we don't have enough information
                    if len(stafflines) < 3 or not noteheads:
                        continue
                    
                    # Sort stafflines by vertical position
                    stafflines = sorted(stafflines, key=lambda x: x['top'])
                    
                    # Get clef type for this staff
                    # clef_elements = [e for e in elements if e['class_name'] in ['gClef', 'fClef', 'cClef']]
                    clef_elements = [e for e in elements if 'Clef' in e['class_name']]
                    clef_type = clef_elements[0]['class_name'] if clef_elements else 'unknown'
                    
                    # For each notehead with pitch information
                    for note in noteheads:
                        if all(k in note for k in ['midi_pitch_code', 'pitch_octave', 'normalized_pitch_step']):
                            # Calculate vertical position relative to stafflines
                            note_center = note['top'] + note['height'] / 2
                            
                            # Find the closest staffline
                            closest_line_idx = min(range(len(stafflines)), 
                                                    key=lambda i: abs(stafflines[i]['top'] - note_center))
                            
                            closest_staffline = stafflines[closest_line_idx]
                            distance = note_center - closest_staffline['top']
                            
                            # Calculate which line/space this corresponds to
                            staff_spacing = 0
                            if len(stafflines) >= 2:
                                staff_spacing = (stafflines[-1]['top'] - stafflines[0]['top']) / (len(stafflines) - 1)
                            
                            # Determine if it's on a line or in a space
                            on_line = abs(distance) < (staff_spacing * 0.25)
                            position_type = "line" if on_line else "space"
                            
                            # Normalize position into standardized nomenclature (L1, S1, etc.)
                            if 'line_pos' in note:
                                standardized_position = note['line_pos']
                            else:
                                # Calculate based on relative position - this is approximate
                                rel_position = distance / staff_spacing if staff_spacing > 0 else 0
                                line_num = closest_line_idx + 1  # 1-based indexing for lines
                                
                                if on_line:
                                    standardized_position = f"L{line_num}"
                                else:
                                    if distance > 0:
                                        # Space below the line
                                        standardized_position = f"S{line_num}"
                                    else:
                                        # Space above the line
                                        standardized_position = f"S{line_num-1}"
                            
                            # Save data
                            pitch_data.append({
                                'midi_pitch': note['midi_pitch_code'],
                                'pitch_octave': note['pitch_octave'],
                                'pitch_step': note['normalized_pitch_step'],
                                'line_pos': standardized_position if 'line_pos' in note else standardized_position,
                                'clef_type': clef_type,
                                'relative_distance': distance,
                                'staff_spacing': staff_spacing,
                                'position_type': position_type,
                            })
            
            except Exception as e:
                print(f"Error processing {xml_file} for pitch analysis: {e}")
        
        # Analyze and visualize pitch-position relationships
        if pitch_data:
            df = pd.DataFrame(pitch_data)
            
            print("\n===== PITCH TO STAFF POSITION ANALYSIS =====")
            print(f"Total notes analyzed: {len(df)}")
            
            # Group by line position and clef to find most common pitch
            grouped = df.groupby(['clef_type', 'line_pos'])
            
            position_to_pitch = {}
            
            print("\nMost common pitches by position:")
            for (clef, pos), group in grouped:
                pitch_counts = group['midi_pitch'].value_counts()
                most_common_pitch = pitch_counts.idxmax() if not pitch_counts.empty else None
                
                if most_common_pitch:
                    # Convert MIDI pitch to note name
                    octave = most_common_pitch // 12 - 1
                    note_idx = most_common_pitch % 12
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    note_name = f"{note_names[note_idx]}{octave}"
                    
                    # FIX HERE: Use string key format instead of tuple for JSON compatibility
                    key = f"{clef}_{pos}"  # Convert tuple (clef, pos) to string "clef_pos"
                    position_to_pitch[key] = {
                        'midi_pitch': most_common_pitch,
                        'note_name': note_name,
                        'count': int(pitch_counts.max()),
                        'percentage': float(pitch_counts.max() / len(group) * 100),
                        'clef': clef,  # Store clef and position in the value for reference
                        'position': pos
                    }
                    
                    print(f"  {clef}, {pos}: {note_name} (MIDI {most_common_pitch}), {pitch_counts.max()}/{len(group)} notes ({pitch_counts.max()/len(group)*100:.1f}%)")
            
            # Plot pitch distribution by line position for most common clef
            most_common_clef = df['clef_type'].value_counts().idxmax()
            clef_df = df[df['clef_type'] == most_common_clef]
            
            # Sort line positions in a sensible order
            def line_pos_sort_key(pos):
                if not isinstance(pos, str):
                    return (3, 0)
                match = re.match(r"([LS])(-?\d+)", pos)
                if not match:
                    return (3, 0)
                prefix, num = match.groups()
                # S prefix comes before L
                prefix_val = 0 if prefix == 'S' else 1
                return (prefix_val, int(num))
            
            positions = sorted(clef_df['line_pos'].unique(), key=line_pos_sort_key)
            
            plt.figure(figsize=(12, 8))
            for i, pos in enumerate(positions):
                pos_data = clef_df[clef_df['line_pos'] == pos]['midi_pitch']
                if len(pos_data) > 0:
                    plt.scatter(pos_data, [i] * len(pos_data), alpha=0.2, s=5)
                    plt.text(-5, i, pos, ha='right', va='center')
            
            plt.title(f'Pitch Distribution by Staff Position ({most_common_clef})')
            plt.xlabel('MIDI Pitch')
            plt.ylabel('Staff Position')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('pitch_by_position.png')
            plt.close()
            
            return {
                'total_notes': len(df),
                'position_to_pitch': position_to_pitch,
                'most_common_clef': most_common_clef
            }
        
        return None

    # Fix for the main function to ensure all dictionaries are JSON serializable
    def main(data_directory):
        """
        Main function to run comprehensive staff integrity analysis
        
        Args:
            data_directory (str): Path to the directory containing XML files
        """
        # Initialize the Staff Integrity Checker
        checker = StaffIntegrityChecker(data_directory)
        
        # Process all XML files in the directory
        checker.process_all_files()
        
        # Run comprehensive analyses
        print("\n===== COMPREHENSIVE STAFF ANALYSIS =====")
        
        # Staff Completeness Analysis
        print("\n1. Staff Completeness Analysis")
        completeness_results = checker.analyze_staff_completeness()
        
        # Staffline Naming Patterns
        print("\n2. Staffline Naming Pattern Analysis")
        naming_patterns = checker.analyze_staffline_naming()
        
        # Staff Spacing Analysis
        print("\n3. Staff Spacing Analysis")
        spacing_results = checker.analyze_staff_spacing()
        
        # Pitch Line Positions Analysis
        print("\n4. Pitch Line Positions Analysis")
        pitch_results = checker.analyze_pitch_line_positions()
        
        # Optional: Save results to a JSON file for further analysis
        import json
        from datetime import datetime
        
        # Ensure all data is JSON serializable
        def json_serialize(obj):
            """Helper function to make all values JSON serializable"""
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {str(k): json_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [json_serialize(item) for item in obj]
            else:
                return obj
        
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'data_directory': str(data_directory),  # Ensure Path objects are converted to strings
            'completeness': json_serialize(completeness_results),
            'naming_patterns': json_serialize(naming_patterns),
            'spacing_analysis': json_serialize(spacing_results),
            'pitch_analysis': json_serialize(pitch_results)
        }
        
        with open('staff_integrity_analysis_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("\nAnalysis complete. Results saved to staff_integrity_analysis_results.json")
        
        return results_summary

    
class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types properly
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        return super(NumpyJSONEncoder, self).default(obj)

# Replace the original main function with this fixed version
def main(data_directory):
    """
    Main function to run comprehensive staff integrity analysis
    
    Args:
        data_directory (str): Path to the directory containing XML files
    """
    # Initialize the Staff Integrity Checker
    checker = StaffIntegrityChecker(data_directory)
    
    # Process all XML files in the directory
    checker.process_all_files()
    
    # Run comprehensive analyses
    print("\n===== COMPREHENSIVE STAFF ANALYSIS =====")
    
    # Staff Completeness Analysis
    print("\n1. Staff Completeness Analysis")
    completeness_results = checker.analyze_staff_completeness()
    
    # Staffline Naming Patterns
    print("\n2. Staffline Naming Pattern Analysis")
    naming_patterns = checker.analyze_staffline_naming()
    
    # Staff Spacing Analysis
    print("\n3. Staff Spacing Analysis")
    spacing_results = checker.analyze_staff_spacing()
    
    # Pitch Line Positions Analysis
    print("\n4. Pitch Line Positions Analysis")
    pitch_results = checker.analyze_pitch_line_positions()
    
    # Create results summary
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_directory': data_directory,
        'completeness': completeness_results,
        'naming_patterns': naming_patterns,
        'spacing_analysis': spacing_results,
        'pitch_analysis': pitch_results
    }
    
    # Save results to a JSON file using custom encoder
    with open('staff_integrity_analysis_results.json', 'w') as f:
        json.dump(results_summary, f, cls=NumpyJSONEncoder, indent=2)
    
    print("\nAnalysis complete. Results saved to staff_integrity_analysis_results.json")
    
    return results_summary

# Example usage
if __name__ == '__main__':
    import sys
    
    # Check if a directory is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <data_directory>")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    main(data_directory)