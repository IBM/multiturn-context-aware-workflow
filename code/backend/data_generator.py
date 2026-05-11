"""
Workflow Data Generator for Multiturn AI Agent Demo
Generates synthetic operational workflow documents with structured steps,
dependencies, required inputs, and completion criteria.
"""

import json
import uuid
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import random


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow"""
    step_number: int
    description: str
    required_inputs: List[str]
    dependencies: List[int]  # Step numbers that must be completed first
    completion_criteria: str
    estimated_duration: str
    api_operation: Optional[str] = None  # Backend API operation to execute


@dataclass
class WorkflowDocument:
    """Represents a complete workflow document"""
    id: str
    type: str
    title: str
    category: str
    description: str
    steps: List[WorkflowStep]
    estimated_duration: str
    required_permissions: List[str]
    tags: List[str]
    created_date: str
    last_updated: str


class WorkflowDataGenerator:
    """Generates synthetic workflow data for the multiturn agent demo"""
    
    def __init__(self):
        self.workflows: List[WorkflowDocument] = []
        self.permissions = [
            "read_metrics", "write_config", "restart_services", 
            "view_logs", "execute_commands", "manage_users",
            "deploy_applications", "modify_network", "access_database",
            "security_admin", "maintenance_user", "work_order_create"
        ]
    
    def add_maintenance_work_order_workflow(self):
        """Add the Create Maintenance Work Order workflow for Scenario 3"""
        workflow = WorkflowDocument(
            id="workflow_maintenance_001",
            type="workflow",
            title="Create Maintenance Work Order",
            category="maintenance",
            description="Create a new maintenance work order for equipment or assets. This workflow collects required information and submits the work order to the maintenance system.",
            steps=[
                WorkflowStep(
                    step_number=1,
                    description="Collect work order details",
                    required_inputs=["asset_id", "maintenance_type", "priority", "scheduled_date"],
                    dependencies=[],
                    completion_criteria="All required information collected",
                    estimated_duration="5 minutes",
                    api_operation="create_work_order"
                )
            ],
            estimated_duration="5 minutes",
            required_permissions=["maintenance_user", "work_order_create"],
            tags=["maintenance", "work_order", "asset_management"],
            created_date="2026-01-15T00:00:00Z",
            last_updated="2026-01-15T00:00:00Z"
        )
        self.workflows.append(workflow)
    
    def generate_operational_procedures(self, count: int = 300):
        """Generate operational procedure workflows"""
        procedure_templates = [
            {
                "title": "Server Restart Procedure",
                "category": "operations",
                "steps": [
                    ("Verify server status", ["server_name"], [], "Status verified", "2 minutes", "check_status"),
                    ("Notify stakeholders", [], [1], "Notifications sent", "5 minutes", "send_notification"),
                    ("Stop running services", [], [2], "Services stopped", "3 minutes", "stop_services"),
                    ("Restart server", [], [3], "Server restarted", "5 minutes", "restart_server"),
                    ("Verify services are running", [], [4], "Services verified", "3 minutes", "verify_services")
                ]
            },
            {
                "title": "Database Backup Procedure",
                "category": "maintenance",
                "steps": [
                    ("Check database connectivity", ["database_name"], [], "Connection verified", "1 minute", "check_connection"),
                    ("Verify backup location", ["backup_path"], [1], "Location verified", "2 minutes", "verify_location"),
                    ("Initiate backup", [], [2], "Backup started", "1 minute", "start_backup"),
                    ("Monitor backup progress", [], [3], "Backup completed", "30 minutes", "monitor_backup"),
                    ("Verify backup integrity", [], [4], "Integrity verified", "5 minutes", "verify_backup")
                ]
            },
            {
                "title": "Security Patch Deployment",
                "category": "security",
                "steps": [
                    ("Identify target systems", ["system_list"], [], "Systems identified", "5 minutes", "identify_systems"),
                    ("Download patch files", ["patch_id"], [1], "Patches downloaded", "10 minutes", "download_patches"),
                    ("Test patch in staging", [], [2], "Testing complete", "20 minutes", "test_patch"),
                    ("Deploy to production", [], [3], "Deployment complete", "15 minutes", "deploy_patch"),
                    ("Verify patch installation", [], [4], "Verification complete", "10 minutes", "verify_patch")
                ]
            }
        ]
        
        for i in range(count):
            template = random.choice(procedure_templates)
            workflow_id = f"proc-{uuid.uuid4()}"
            
            steps = []
            for idx, (desc, inputs, deps, criteria, duration, api_op) in enumerate(template["steps"], 1):
                steps.append(WorkflowStep(
                    step_number=idx,
                    description=desc,
                    required_inputs=inputs,
                    dependencies=deps,
                    completion_criteria=criteria,
                    estimated_duration=duration,
                    api_operation=api_op
                ))
            
            total_duration = sum(int(s.estimated_duration.split()[0]) for s in steps)
            
            workflow = WorkflowDocument(
                id=workflow_id,
                type="operational_procedure",
                title=f"{template['title']} - Variant {i % 10 + 1}",
                category=template["category"],
                description=f"Standard operating procedure for {template['title'].lower()}",
                steps=steps,
                estimated_duration=f"{total_duration} minutes",
                required_permissions=random.sample(self.permissions, k=random.randint(2, 4)),
                tags=[template["category"], "operational", "standard"],
                created_date=(datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                last_updated=datetime.now().isoformat()
            )
            
            self.workflows.append(workflow)
    
    def generate_task_plans(self, count: int = 250):
        """Generate task plan workflows"""
        task_templates = [
            {
                "title": "Server Maintenance Window",
                "category": "maintenance",
                "steps": [
                    ("Schedule maintenance window", ["start_time", "duration"], [], "Window scheduled", "5 minutes", "schedule_window"),
                    ("Notify affected users", [], [1], "Notifications sent", "10 minutes", "send_notifications"),
                    ("Create backup of critical data", ["backup_location"], [2], "Backup completed", "30 minutes", "create_backup"),
                    ("Apply system updates", [], [3], "Updates applied", "45 minutes", "apply_updates"),
                    ("Restart services", [], [4], "Services restarted", "10 minutes", "restart_services"),
                    ("Verify system functionality", [], [5], "Verification complete", "15 minutes", "verify_system"),
                    ("Close maintenance window", [], [6], "Window closed", "5 minutes", "close_window")
                ]
            },
            {
                "title": "New Asset Onboarding",
                "category": "configuration",
                "steps": [
                    ("Register asset in CMDB", ["asset_id", "asset_type", "location"], [], "Asset registered", "5 minutes", "register_asset"),
                    ("Configure monitoring", ["monitoring_profile"], [1], "Monitoring configured", "10 minutes", "setup_monitoring"),
                    ("Install required software", ["software_list"], [2], "Software installed", "30 minutes", "install_software"),
                    ("Configure network settings", ["ip_address", "subnet"], [3], "Network configured", "15 minutes", "configure_network"),
                    ("Run validation tests", [], [4], "Tests passed", "20 minutes", "run_validation"),
                    ("Document configuration", [], [5], "Documentation complete", "10 minutes", "create_documentation")
                ]
            },
            {
                "title": "Incident Response",
                "category": "troubleshooting",
                "steps": [
                    ("Acknowledge incident", ["incident_id"], [], "Incident acknowledged", "1 minute", "acknowledge_incident"),
                    ("Assess impact and priority", [], [1], "Assessment complete", "5 minutes", "assess_impact"),
                    ("Identify affected systems", [], [2], "Systems identified", "10 minutes", "identify_systems"),
                    ("Implement temporary workaround", ["workaround_description"], [3], "Workaround applied", "15 minutes", "apply_workaround"),
                    ("Investigate root cause", [], [4], "Root cause found", "30 minutes", "investigate_cause"),
                    ("Implement permanent fix", ["fix_description"], [5], "Fix implemented", "20 minutes", "implement_fix"),
                    ("Verify resolution", [], [6], "Resolution verified", "10 minutes", "verify_resolution"),
                    ("Close incident", [], [7], "Incident closed", "5 minutes", "close_incident")
                ]
            }
        ]
        
        for i in range(count):
            template = random.choice(task_templates)
            workflow_id = f"task-{uuid.uuid4()}"
            
            steps = []
            for idx, (desc, inputs, deps, criteria, duration, api_op) in enumerate(template["steps"], 1):
                steps.append(WorkflowStep(
                    step_number=idx,
                    description=desc,
                    required_inputs=inputs,
                    dependencies=deps,
                    completion_criteria=criteria,
                    estimated_duration=duration,
                    api_operation=api_op
                ))
            
            total_duration = sum(int(s.estimated_duration.split()[0]) for s in steps)
            
            workflow = WorkflowDocument(
                id=workflow_id,
                type="task_plan",
                title=f"{template['title']} - Plan {i % 10 + 1}",
                category=template["category"],
                description=f"Detailed task plan for {template['title'].lower()}",
                steps=steps,
                estimated_duration=f"{total_duration} minutes",
                required_permissions=random.sample(self.permissions, k=random.randint(3, 5)),
                tags=[template["category"], "task", "multi-step"],
                created_date=(datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                last_updated=datetime.now().isoformat()
            )
            
            self.workflows.append(workflow)
    
    def generate_troubleshooting_guides(self, count: int = 200):
        """Generate troubleshooting guide workflows"""
        for i in range(count):
            workflow_id = f"guide-{uuid.uuid4()}"
            
            # Simplified troubleshooting guides with 3-5 steps
            num_steps = random.randint(3, 5)
            steps = []
            
            for step_num in range(1, num_steps + 1):
                steps.append(WorkflowStep(
                    step_number=step_num,
                    description=f"Troubleshooting step {step_num}: Diagnose and resolve issue",
                    required_inputs=["issue_description"] if step_num == 1 else [],
                    dependencies=list(range(1, step_num)),
                    completion_criteria=f"Step {step_num} completed",
                    estimated_duration=f"{random.randint(5, 15)} minutes",
                    api_operation=f"troubleshoot_step_{step_num}" if step_num > 1 else None
                ))
            
            workflow = WorkflowDocument(
                id=workflow_id,
                type="troubleshooting_guide",
                title=f"Troubleshooting Guide {i + 1}",
                category=random.choice(["performance", "troubleshooting", "maintenance"]),
                description=f"Step-by-step troubleshooting guide for common issues",
                steps=steps,
                estimated_duration=f"{sum(int(s.estimated_duration.split()[0]) for s in steps)} minutes",
                required_permissions=random.sample(self.permissions, k=random.randint(2, 3)),
                tags=["troubleshooting", "guide", "diagnostic"],
                created_date=(datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                last_updated=datetime.now().isoformat()
            )
            
            self.workflows.append(workflow)
    
    def generate_configuration_procedures(self, count: int = 150):
        """Generate configuration procedure workflows"""
        for i in range(count):
            workflow_id = f"config-{uuid.uuid4()}"
            
            steps = [
                WorkflowStep(
                    step_number=1,
                    description="Review current configuration",
                    required_inputs=["system_name"],
                    dependencies=[],
                    completion_criteria="Configuration reviewed",
                    estimated_duration="5 minutes",
                    api_operation="get_config"
                ),
                WorkflowStep(
                    step_number=2,
                    description="Backup existing configuration",
                    required_inputs=[],
                    dependencies=[1],
                    completion_criteria="Backup created",
                    estimated_duration="3 minutes",
                    api_operation="backup_config"
                ),
                WorkflowStep(
                    step_number=3,
                    description="Apply new configuration",
                    required_inputs=["config_parameters"],
                    dependencies=[2],
                    completion_criteria="Configuration applied",
                    estimated_duration="10 minutes",
                    api_operation="apply_config"
                ),
                WorkflowStep(
                    step_number=4,
                    description="Validate configuration changes",
                    required_inputs=[],
                    dependencies=[3],
                    completion_criteria="Validation passed",
                    estimated_duration="5 minutes",
                    api_operation="validate_config"
                )
            ]
            
            workflow = WorkflowDocument(
                id=workflow_id,
                type="configuration_procedure",
                title=f"Configuration Procedure {i + 1}",
                category="configuration",
                description="Standard configuration change procedure",
                steps=steps,
                estimated_duration="23 minutes",
                required_permissions=["write_config", "restart_services"],
                tags=["configuration", "change", "standard"],
                created_date=(datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                last_updated=datetime.now().isoformat()
            )
            
            self.workflows.append(workflow)
    
    def generate_inspection_checklists(self, count: int = 100):
        """Generate inspection checklist workflows"""
        for i in range(count):
            workflow_id = f"checklist-{uuid.uuid4()}"
            
            num_checks = random.randint(5, 8)
            steps = []
            
            for step_num in range(1, num_checks + 1):
                steps.append(WorkflowStep(
                    step_number=step_num,
                    description=f"Inspection checkpoint {step_num}",
                    required_inputs=["asset_id"] if step_num == 1 else [],
                    dependencies=list(range(1, step_num)),
                    completion_criteria=f"Checkpoint {step_num} verified",
                    estimated_duration="3 minutes",
                    api_operation=f"verify_checkpoint_{step_num}"
                ))
            
            workflow = WorkflowDocument(
                id=workflow_id,
                type="inspection_checklist",
                title=f"Inspection Checklist {i + 1}",
                category="inspection",
                description="Standard inspection checklist for asset verification",
                steps=steps,
                estimated_duration=f"{num_checks * 3} minutes",
                required_permissions=["read_metrics", "view_logs"],
                tags=["inspection", "checklist", "verification"],
                created_date=(datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                last_updated=datetime.now().isoformat()
            )
            
            self.workflows.append(workflow)
    
    def generate_all_data(self):
        """Generate all workflow types"""
        print("Adding Create Maintenance Work Order workflow (for Scenario 3)...")
        self.add_maintenance_work_order_workflow()
        
        print("Generating operational procedures...")
        self.generate_operational_procedures(300)
        
        print("Generating task plans...")
        self.generate_task_plans(250)
        
        print("Generating troubleshooting guides...")
        self.generate_troubleshooting_guides(200)
        
        print("Generating configuration procedures...")
        self.generate_configuration_procedures(150)
        
        print("Generating inspection checklists...")
        self.generate_inspection_checklists(100)
        
        print(f"Total workflows generated: {len(self.workflows)}")
    
    def save_to_json(self, filepath: str = "../data/workflow_dataset.json"):
        """Save generated workflows to JSON file"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert dataclasses to dicts
        data = [asdict(workflow) for workflow in self.workflows]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} workflows to {filepath}")


if __name__ == "__main__":
    generator = WorkflowDataGenerator()
    generator.generate_all_data()
    generator.save_to_json()
    print("Workflow data generation complete!")

# Made with Bob
