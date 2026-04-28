import { Component } from '@angular/core';
import {
  GeneratedPatient,
  VanillaGanService,
} from './services/vanilla-gan.service';

interface ArchLayer {
  name: string;
  detail: string;
}

interface MetricFigure {
  file: string;
  title: string;
  description: string;
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent {
  readonly latentDim = 128;
  readonly hiddenWidth = 256;
  readonly epochs = 200;
  readonly batchSize = 500;
  readonly optimizer = 'Adam (lr=2e-4, betas=(0.5, 0.999))';
  readonly loss = 'Binary Cross-Entropy (non-saturating G loss)';
  readonly dataset = 'MSK-IMPACT 50k — 80% train / 20% test';

  readonly generatorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'z ∈ ℝ^128 (Gaussian noise)' },
    { name: 'Linear', detail: '128 → 256' },
    { name: 'LeakyReLU', detail: 'negative slope = 0.2' },
    { name: 'Linear', detail: '256 → 256' },
    { name: 'LeakyReLU', detail: 'negative slope = 0.2' },
    { name: 'Linear', detail: '256 → 256' },
    { name: 'LeakyReLU', detail: 'negative slope = 0.2' },
    { name: 'Linear', detail: '256 → n_features' },
    { name: 'Output', detail: 'synthetic patient vector (z-score space)' },
  ];

  readonly discriminatorLayers: ArchLayer[] = [
    { name: 'Input', detail: 'patient vector ∈ ℝ^n_features' },
    { name: 'Linear', detail: 'n_features → 256' },
    { name: 'LeakyReLU + Dropout', detail: 'slope = 0.2, p = 0.3' },
    { name: 'Linear', detail: '256 → 256' },
    { name: 'LeakyReLU + Dropout', detail: 'slope = 0.2, p = 0.3' },
    { name: 'Linear', detail: '256 → 1' },
    { name: 'Sigmoid', detail: 'P(real)' },
  ];

  readonly figures: MetricFigure[] = [
    {
      file: 'cindex_brier_comparison.png',
      title: 'C-Index & Brier Score Comparison',
      description:
        'Compares survival models trained on real vs. GAN-synthetic data using Concordance Index (higher = better risk ranking) and Brier Score (lower = better calibrated predictions). If the synthetic-trained model tracks close to the real-trained baseline, the GAN preserves the survival signal needed for downstream prediction.',
    },
    {
      file: 'metric_heatmap.png',
      title: 'Per-Feature Fidelity Heatmap',
      description:
        'Heatmap of fidelity metrics (e.g. KS distance, Wasserstein, marginal-mean error) computed per column. Darker / lower-error cells indicate marginals the GAN reproduces faithfully; bright cells flag features where the synthetic distribution drifts from the real one — a signal of mode collapse or biased sampling on those columns.',
    },
    {
      file: 'radar_summary.png',
      title: 'Radar Summary of Quality Dimensions',
      description:
        'Single-glance radar that scores the GAN across statistical fidelity, downstream utility, privacy, and diversity axes. A balanced large polygon means the model is well-rounded; sharp inward dents on any axis expose the weakest dimension (commonly diversity for vanilla GANs prone to mode collapse).',
    },
    {
      file: 'subgroup_km_grid.png',
      title: 'Subgroup Kaplan–Meier Curves',
      description:
        'Grid of Kaplan–Meier survival curves for clinically meaningful subgroups (e.g. cancer type, sex, MSI status), real vs. synthetic. Overlapping curves mean the GAN preserves subgroup-level survival dynamics; large gaps reveal that the synthetic cohort misrepresents how a subgroup actually progresses over time.',
    },
    {
      file: 'subgroup_metric_heatmap.png',
      title: 'Subgroup Metric Heatmap',
      description:
        'Survival-metric error (e.g. C-index gap, log-rank p-value) broken down by subgroup. It exposes whether the GAN performs uniformly across the population or systematically fails on minority subgroups — a key fairness check before using synthetic data for downstream research.',
    },
  ];

  generating = false;
  generated: GeneratedPatient | null = null;
  generateError: string | null = null;

  constructor(private vanillaGan: VanillaGanService) {}

  generatePatient(): void {
    this.generating = true;
    this.generateError = null;
    this.vanillaGan.generate().subscribe({
      next: (patient) => {
        this.generated = patient;
        this.generating = false;
      },
      error: (err) => {
        this.generated = null;
        this.generateError =
          err?.message ??
          'Could not reach the GAN backend. Make sure backend/app.py is running on :5000.';
        this.generating = false;
      },
    });
  }

  formatValue(column: string, raw: number): string {
    if (Number.isInteger(raw)) {
      return raw.toString();
    }
    return raw.toFixed(3);
  }

  formatLatent(values: number[]): string {
    return values.map((v) => v.toFixed(3)).join(', ');
  }
}
