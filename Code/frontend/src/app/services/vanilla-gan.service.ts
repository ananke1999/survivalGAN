import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface GeneratedPatient {
  columns: string[];
  values: Record<string, number>;
  latent_preview: number[];
  raw_preview: number[];
}

@Injectable({ providedIn: 'root' })
export class VanillaGanService {
  private readonly endpoint = '/api/generate';

  constructor(private http: HttpClient) {}

  generate(): Observable<GeneratedPatient> {
    return this.http.post<GeneratedPatient>(this.endpoint, {});
  }
}
